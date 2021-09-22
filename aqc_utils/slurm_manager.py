import hashlib
import pickle
from contextlib import suppress

import appdirs
import pymongo

from aqc_utils.db_functions import *
from aqc_utils.gaussian_input_generator import *
from aqc_utils.helper_functions import *
from aqc_utils.openbabel_functions import *

logger = logging.getLogger(__name__)


class slurm_manager(object):
    """Slurm manager class."""

    def __init__(self, user, host):
        """Initialize slurm manager and load the cache file.

        :param user: username at remote host
        :type user: str
        :param host: remote host name
        :type host: str
        """

        # set workdir and cache file
        self.workdir = appdirs.user_data_dir(appauthor="autoqchem", appname=host.split('.')[0])
        self.cache_file = os.path.join(self.workdir, "slurm_manager.pkl")
        os.makedirs(self.workdir, exist_ok=True)

        self.jobs = {}  # jobs under management

        # load jobs under management from cache_file (suppress exceptions, no file, empty file, etc.)
        with suppress(Exception):  # TODO this isn't very safe because may ignore important exceptions
            with open(self.cache_file, 'rb') as cf:
                self.jobs = pickle.load(cf)

        self.host = host
        self.user = user
        self.password = 'Fisk5oc%'
        self.remote_dir = f"/home/{self.user}/gaussian"
        self.connection = None

    def connect(self) -> None:
        """Connect to remote host."""

        create_new_connection = False
        # check if connection already exists
        if self.connection is not None:
            # check if it went stale
            if not self.connection.is_connected:
                logger.info(f"Connection got disconnected, reconnecting.")
                self.connection.close()
                self.connection = None
                create_new_connection = True
        else:
            logger.info(f"Creating connection to {self.host} as {self.user}")
            create_new_connection = True
        if create_new_connection:
            self.connection = ssh_connect(self.host, self.user)
            self.connection.run(f"mkdir -p {self.remote_dir}")
            logger.info(f"Connected to {self.host} as {self.user} is {self.connection.is_connected}.")          

    def create_jobs_for_molecule(self,
                                 molecule,
                                 workflow_type="equilibrium",
                                 theory="b3lyp",
                                 light_basis_set="6-31G*",
                                 heavy_basis_set="LANL2DZ",
                                 generic_basis_set="genecp",
                                 max_light_atomic_number=36,
                                 wall_time='1000:00:00') -> None:
        """Generate slurm jobs for a molecule. Gaussian input files are also generated.

        :param molecule: molecule object
        :type molecule: molecule
        :param workflow_type: Gaussian workflow type, allowed types are: 'equilibrium' or 'transition_state'
        :type workflow_type: str
        :param theory: Gaussian theory functional
        :type theory: str
        :param light_basis_set: basis set to use for light elements
        :type light_basis_set: str
        :param heavy_basis_set: basis set to use for heavy elements
        :type heavy_basis_set: str
        :param generic_basis_set: basis set to use for generic elements
        :type generic_basis_set: str
        :param max_light_atomic_number: maximum atomic number for light elements
        :type max_light_atomic_number: int
        :param wall_time: wall time of the job in HH:MM:SS format
        :type wall_time: str
        """

        # create gaussian files
        molecule_workdir = os.path.join(self.workdir, molecule.fs_name)
        print(molecule_workdir)
        gig = gaussian_input_generator(molecule, workflow_type, molecule_workdir, theory, light_basis_set,
                                       heavy_basis_set, generic_basis_set, max_light_atomic_number)
        gaussian_config = {'theory': theory,
                           'light_basis_set': light_basis_set,
                           'heavy_basis_set': heavy_basis_set,
                           'generic_basis_set': generic_basis_set,
                           'max_light_atomic_number': max_light_atomic_number}

        gig.create_gaussian_files()

        # create slurm files
        for gjf_file in glob.glob(f"{molecule_workdir}/*.com"):

            base_name = os.path.basename(os.path.splitext(gjf_file)[0])
            print(base_name)
            self._create_slurm_file_from_gaussian_file(base_name, molecule_workdir, wall_time)
            # create job structure
            job = slurm_job(can=molecule.can,
                            conformation=int(base_name.split("_conf_")[1]),
                            max_num_conformers=gig.molecule.max_num_conformers,
                            tasks=gig.tasks,
                            config=gaussian_config,
                            job_id=-1,  # job_id (not assigned yet)
                            directory=gig.directory,  # filesystem path
                            base_name=base_name,  # filesystem basename
                            status=slurm_status.created,
                            n_submissions=0,
                            n_success_tasks=0)  # status

            # create a key for the job
            key = hashlib.md5((job.can + str(job.conformation) +
                               str(job.max_num_conformers) + ','.join(map(str, job.tasks))).encode()).hexdigest()

            # check if a job like that already exists
            if key in self.jobs:  # a job like that is already present:
                logger.warning(f"A job with exactly the same parameters, molecule {job.can}, conformation "
                               f"{job.conformation}, workflow {job.tasks} already exists. "
                               f"Not creating a duplicate")
                continue

            self.jobs[key] = job  # add job to bag
        self._cache()

    def submit_jobs(self) -> None:
        """Submit jobs that have status 'created' to remote host."""

        jobs = self.get_jobs(slurm_status.created)
        logger.info(f"Submitting {len(jobs)} jobs.")
        self.submit_jobs_from_jobs_dict(jobs)

    def submit_jobs_from_jobs_dict(self, jobs) -> None:
        """Submit jobs to remote host.

        :param jobs: dictionary of jobs to submit
        :type jobs: dict
        """

        # check if there are any jobs to be submitted
        if jobs:
            # get or create connection
            self.connect()

            # check if jobs are in status created or failed
            for name, job in jobs.items():
                # copy .sh and .gjf file to remote_dir
                self.connection.put(f"{job.directory}/{job.base_name}.sh", self.remote_dir)
                self.connection.put(f"{job.directory}/{job.base_name}.com", self.remote_dir)

                with self.connection.cd(self.remote_dir):
                    ret = self.connection.run(f"sbatch {self.remote_dir}/{job.base_name}.sh", hide=True)
                    job.job_id = re.search("job\s*(\d+)\n", ret.stdout).group(1)
                    job.status = slurm_status.submitted
                    job.n_submissions = job.n_submissions + 1
                    logger.info(f"Submitted job {name}, job_id: {job.job_id}.")

            self._cache()

    def retrieve_jobs(self) -> None:
        """Retrieve finished jobs from remote host and check which finished succesfully and which failed."""

        ids_to_check = [j.job_id for j in self.get_jobs(slurm_status.submitted).values()]
        if not ids_to_check:
            logger.info(f"There are no jobs submitted to cluster. Nothing to retrieve.")
            return

        # get or create connection
        self.connect()

        # retrieve job ids that are running on the server
        ret = self.connection.run(f"squeue -u {self.user} -o %A,%T", hide=True)
        user_running_ids = [s.split(',')[0] for s in ret.stdout.splitlines()[1:]]
        running_ids = [id for id in user_running_ids if id in ids_to_check]
        finished_ids = [id for id in ids_to_check if id not in running_ids]

        logger.info(f"There are {len(running_ids)} running/pending jobs, {len(finished_ids)} finished jobs.")

        # get finished jobs
        finished_jobs = {name: job for name, job in self.jobs.items() if job.job_id in finished_ids}
        done_jobs = 0

        if finished_jobs:
            logger.info(f"Retrieving log files of finished jobs.")
            for job in finished_jobs.values():
                status = self._retrieve_single_job(job)
                if status.value == slurm_status.done.value:
                    done_jobs += 1

            self._cache()
            logger.info(f"{done_jobs} jobs finished successfully (all Gaussian steps finished normally)."
                        f" {len(finished_jobs) - done_jobs} jobs failed.")

    def _retrieve_single_job(self, job) -> slurm_status:
        """Retrieve single job from remote host and check its status

        :param job: job
        :return: :py:meth:`~helper_classes.helper_classes.slurm_status`, resulting status
        """

        try:  # try to fetch the file
            log_file = self.connection.get(f"{self.remote_dir}/{job.base_name}.log",
                                           local=f"{job.directory}/{job.base_name}.log")

            # initialize the log extractor, it will try to read basic info from the file
            le = gaussian_log_extractor(log_file.local)
            if len(job.tasks) == le.n_tasks:
                job.status = slurm_status.done
            else:
                try:  # look for more specific exception
                    le.check_for_exceptions()

                except NoGeometryException:
                    job.status = slurm_status.failed
                    logger.warning(
                        f"Job {job.base_name} failed - the log file does not contain geometry. Cannot resubmit.")

                except NegativeFrequencyException:
                    job.status = slurm_status.incomplete
                    logger.warning(
                        f"Job {job.base_name} incomplete - log file contains negative frequencies. Resubmit job.")

                except OptimizationIncompleteException:
                    job.status = slurm_status.incomplete
                    logger.warning(f"Job {job.base_name} incomplete - geometry optimization did not complete.")

                except Exception as e:
                    job.status = slurm_status.failed
                    logger.warning(f"Job {job.base_name} failed with unhandled exception: {e}")

                else:  # no exceptions were thrown, but still the job is incomplete
                    job.status = slurm_status.incomplete
                    logger.warning(f"Job {job.base_name} incomplete.")

        except FileNotFoundError:
            job.status = slurm_status.failed
            logger.warning(f"Job {job.base_name} failed  - could not retrieve log file. Cannot resubmit.")

        # clean up files on the remote site - do not cleanup anything, the /scratch/network cleans
        # up files that are older than 15 days

        return job.status

    def resubmit_incomplete_jobs(self, wall_time="24:59:00") -> None:
        """Resubmit jobs that are incomplete. If the job has failed because the optimization has not completed \
        and a log file has been retrieved, then \
        the last geometry will be used for the next submission. For failed jobs \
         the job input files will need to be fixed manually and submitted using the \
        function :py:meth:`~slurm_manager.slurm_manager.submit_jobs_from_jobs_dict`.\
         Maximum number of allowed submission of the same job is 3.

        :param wall_time: wall time of the job in HH:MM:SS format
        :return: None
        """

        incomplete_jobs = self.get_jobs(slurm_status.incomplete)
        incomplete_jobs_to_resubmit = {}

        if not incomplete_jobs:
            logger.info("There are no incomplete jobs to resubmit.")

        for key, job in incomplete_jobs.items():

            # put a limit on resubmissions
            if job.n_submissions >= 3:
                logger.warning(f"Job {job.base_name} has been already failed 3 times, not submitting again.")
                continue

            job_log = f"{job.directory}/{job.base_name}.log"
            job_gjf = f"{job.directory}/{job.base_name}.com"

            # replace geometry
            le = gaussian_log_extractor(job_log)
            le.get_atom_labels()
            le.get_geometry()
            # old coords block
            with open(job_gjf, "r") as f:
                file_string = f.read()
            old_coords_block = re.search(f"\w+\s+({float_or_int_regex})"
                                         f"\s+({float_or_int_regex})"
                                         f"\s+({float_or_int_regex}).*?\n\n",
                                         file_string, re.DOTALL).group(0)

            # new coords block
            coords = le.geom[list('XYZ')].applymap(lambda x: f"{x:.6f}")
            coords.insert(0, 'Atom', le.labels)
            coords_block = "\n".join(map(" ".join, coords.values)) + "\n\n"

            # make sure they are the same length and replace
            assert len(old_coords_block.splitlines()) == len(coords_block.splitlines())
            file_string = file_string.replace(old_coords_block, coords_block)
            with open(job_gjf, "w") as f:
                f.write(file_string)

            logger.info("Substituting last checked geometry in the new input file.")

            # replacing wall time
            job_sh = f"{job.directory}/{job.base_name}.sh"
            with open(job_sh, "r") as f:
                file_string = f.read()
            old_wall_time = re.search(f"#SBATCH -t (\d\d:\d\d:\d\d)", file_string).group(1)
            file_string = file_string.replace(old_wall_time, wall_time)
            with open(job_sh, "w") as f:
                f.write(file_string)
            convert_crlf_to_lf(job_sh)

            logger.info(f"Substituting wall_time with new value: {wall_time}")
            incomplete_jobs_to_resubmit[key] = job

        self.submit_jobs_from_jobs_dict(incomplete_jobs_to_resubmit)

    def upload_done_molecules_to_db(self, tags, cls="", subcls="",
                                    type="", subtype="", RMSD_threshold=0.01, symmetry=True) -> None:

        """Upload done molecules to db. Molecules are considered done when all jobs for a given \
         smiles are in 'done' status. The conformers are deduplicated and uploaded to database using a metadata tag.

        :param tag: metadata tag to use for these molecules in the database
        :type tag: str
        :param RMSD_threshold: RMSD threshold (in Angstroms) to use when deduplicating multiple conformers \
        after Gaussian has found optimal geometry
        :type RMSD_threshold: float
        :param symmetry: if True symmetry is taken into account when comparing molecules in OBAlign(symmetry=True)
        :type symmetry: bool
        """

        done_jobs = self.get_jobs(slurm_status.done)
        if not done_jobs:
            logger.info("There are no jobs in done status. Exitting.")
            return

        # check if there are molecules with all jobs done
        dfj = self.get_job_stats(split_by_can=True)
        dfj_done = dfj[dfj['done'] == dfj.sum(1)]  # only done jobs
        done_cans = dfj_done.index.tolist()

        if not done_cans:
            logger.info("There are no molecules with all jobs done. Exitting.")
            return

        logger.info(f"There are {len(done_cans)} finished molecules {done_cans}.")

        # select jobs for done molecules
        done_can_jobs = self.get_jobs(can=done_cans)
        jobs_df = pd.DataFrame([job.__dict__ for job in done_can_jobs.values()], index=done_can_jobs.keys())

        logger.debug(f"Deduplicating conformers if RMSD < {RMSD_threshold}.")
        meta = {"class": cls, "subclass": subcls, "type": type, "subtype": subtype}

        for (can, tasks, max_n_conf), keys in jobs_df.groupby(["can", "tasks", "max_num_conformers"]).groups.items():

            if len(keys) > 1:
                # deduplicate conformers
                mols = [OBMol_from_done_slurm_job(done_jobs[key]) for key in keys]
                duplicates = deduplicate_list_of_OBMols(mols, RMSD_threshold=RMSD_threshold, symmetry=symmetry)
                logger.info(f"Molecule {can} has {len(duplicates)} / {len(keys)} duplicate conformers.")

                # fetch non-duplicate keys
                can_keys_to_keep = [key for i, key in enumerate(keys) if i not in duplicates]
            else:
                can_keys_to_keep = keys
            self._upload_can_to_db(can, tasks, meta, can_keys_to_keep, tags, max_n_conf)

    def _upload_can_to_db(self, can, tasks, meta, keys, tags, max_conf) -> None:
        """Uploading single molecule conformers to database.

        :param db: database client
        :type db: pymongo.MongoClient
        :param can: canonical smiles
        :type can: str
        :param tasks: tuple of Gaussian tasks
        :type tasks: tuple
        :param keys: list of keys to the self.jobs dictionary to upload
        :type keys: list
        :param tags: metadata tag or tags
        :type tags: str or list
        :param max_conf: max number of conformers used for this molecule
        :type max_conf: int
        """

        # check if the tag(s) are properly provided
        assert isinstance(tags, (str, list))
        if isinstance(tags, str):
            tags = [tags]
        assert all(len(t.strip()) > 0 for t in tags)

        # loop over the conformers
        conformations = []
        logs = []
        configs = []
        for key in keys:
            # fetch job, verify that there are not can issues (just in case)
            job = self.jobs[key]
            assert job.can == can

            # append job configs
            configs.append(job.config)

            # extract descriptors for this conformer from log file
            log = f"{job.directory}/{job.base_name}.log"
            le = gaussian_log_extractor(log)
            # add descriptors to conformations list
            conformations.append(le.get_descriptors())
            logs.append(le.log)

        # compute weights
        free_energies = np.array(
            [Hartree_in_kcal_per_mol * c['descriptors']['G'] for c in conformations])  # in kcal_mol
        free_energies -= free_energies.min()  # to avoid huge exponentials
        weights = np.exp(-free_energies / (k_in_kcal_per_mol_K * T))
        weights /= weights.sum()

        # check that all configs are the same
        assert len(set([config.__repr__() for config in configs])) == 1
        metadata = {'gaussian_config': configs[0], 'gaussian_tasks': tasks, 'max_num_conformers': max_conf}
        metadata.update(meta)

        mol_id = db_upload_molecule(can, tags, metadata, weights, conformations, logs)
        logger.info(f"Uploaded descriptors to DB for smiles: {can}, number of conformers: {len(conformations)},"
                    f" DB molecule id {mol_id}.")
        for key in keys:
            self.jobs[key].status = slurm_status.uploaded
        self._cache()

    def get_jobs(self, status=None, can=None) -> dict:
        """Get a dictionary of jobs, optionally filter by status and canonical smiles.

        :param status: slurm status of the jobs
        :type status: slurm_status
        :param can: canonical smiles of the molecules, single string for one smiles, a list for multiple smiles
        :type can: str or list
        :return: dict
        """

        def match(job, status, can):
            match = True
            if status is not None:
                match = match and job.status.value == status.value
            if can is not None:
                if isinstance(can, str):
                    can = [can]
                match = match and (job.can in can)
            return match

        return {name: job for name, job in self.jobs.items() if match(job, status, can)}

    def get_job_stats(self, split_by_can=False) -> pd.DataFrame:
        """Job stats for jobs currently under management, optionally split by canonical smiles.

        :param split_by_can: if True each canonical smiles will be listed separately
        :type split_by_can: bool
        :return: pandas.core.frame.DataFrame
        """

        df = pd.DataFrame([[v.status.name, v.can] for v in self.jobs.values()], columns=['status', 'can'])
        if split_by_can:
            return df.groupby(['status', 'can']).size().unstack(level=1).fillna(0).astype(int).T
        else:
            return df.groupby('status').size().to_frame('jobs').T

    def remove_jobs(self, jobs) -> None:
        """Remove jobs.

        :param jobs: dictionary of jobs to remove
        :type jobs: dict
        """

        self.connect()
        for name, job in jobs.items():
            logger.debug(f"Removing job {name}.")
            # remove local files
            os.remove(f"{job.directory}/{job.base_name}.sh")  # slurm file
            os.remove(f"{job.directory}/{job.base_name}.com")  # gaussian file
            if os.path.exists(f"{job.directory}/{job.base_name}.log"):
                os.remove(f"{job.directory}/{job.base_name}.log")  # log file
            # remove remote files
            self.connection.run(f"rm -f {self.remote_dir}/slurm-{job.job_id}.out")
            self.connection.run(f"rm -f {self.remote_dir}/{job.base_name}*")
            del self.jobs[name]
        self._cache()

    def squeue(self, summary=True) -> pd.DataFrame:
        """Run 'squeue -u $user' command on the server.

        :param summary: if True only a summary frame is displayed with counts of jobs in each status
        :return: pandas.core.frame.DataFrame
        """

        self.connect()
        if summary:
            ret = self.connection.run(f"squeue -u {self.user} -o %T", hide=True)
            status_series = pd.Series(ret.stdout.splitlines()[1:])
            return status_series.groupby(status_series).size().to_frame("jobs").T
        else:
            ret = self.connection.run(f"squeue -u {self.user}", hide=True)
            data = np.array(list(map(str.split, ret.stdout.splitlines())))
            return pd.DataFrame(data[1:], columns=data[0])

    def _scancel(self) -> None:
        """Run 'scancel -u $user' command on the server."""

        self.connect()
        self.connection.run(f"scancel -u {self.user}")
        self.remove_jobs(self.get_jobs(status=slurm_status.submitted))

    def _cache(self) -> None:
        """save jobs under management and cleanup empty directories"""

        with open(self.cache_file, 'wb') as cf:
            pickle.dump(self.jobs, cf)

        cleanup_empty_dirs(self.workdir)

    def _create_slurm_file_from_gaussian_file(self, base_name, directory, wall_time) -> None:
        """Generate a single slurm submission file based on the Gaussian input file.

        :param base_name: base name of the Gaussian file
        :param directory: directory location of the Gaussian file
        :param wall_time: wall time of the job in HH:MM:SS format
        """
        
        # get information from gaussian file needed for submission
        with open(f"{directory}/{base_name}.com") as f:
            file_string = f.read()

        host = self.host.split(".")[0]

        #n_processors = re.search("nprocshared=(.*?)\n", file_string).group(1)
        n_processors=1
        #constraint = {'della': '\"haswell|skylake\"', 'adroit': '\"skylake\"'}[host]
        g09root = "{g09root}"
        
        output = ""
        output += f"#! /bin/bash\n"
        output += f"#SBATCH -J runG09\n" \
                  f"#SBATCH --ntasks={n_processors}\n" \
                  f"#SBATCH --time {wall_time}\n"
                  #f"#SBATCH --constraint={constraint}\n\n"
        output += f"#SBATCH --output {base_name}.out\n"
        output += f"#SBATCH --mail-type=ALL\n" \
                  f"#SBATCH --mail-user=jules.schleinitz@ens.fr\n\n"
        output += f"#############################################################\n\n"
        output += f"module purge\n" \
                  f"module load gaussian\n\n" \
                  f"source ${g09root}/g09.profile\n\n" \
                  f"g09 {base_name}"
        
        #if host == "adroit":
        #    output += f"module load gaussian/g16\n\n"
        #output += f"input={base_name}\n\n"
        #output += f"# run the code \n" \
        #          f"cd {self.remote_dir}\n" \
        #          f"g16 ${{input}}.gjf\n\n"

        sh_file_path = f"{directory}/{base_name}.sh"
        with open(sh_file_path, "w") as f:
            f.write(output)
        convert_crlf_to_lf(sh_file_path)
        logger.debug(f"Created a Slurm job file in {sh_file_path}")
