#! /bin/bash
#SBATCH -J runG09
#SBATCH --ntasks=1
#SBATCH --time=1000:00:00
#SBATCH --mol_out392
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jules.schleinitz@ens.fr

#########################################################

module purge
module load gaussian

source ${g09root}/g09.profile

g09 mol_inp392
