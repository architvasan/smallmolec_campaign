#!/bin/bash
#PBS -N rtcb_inf
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -q lustre_scaling
#PBS -A Aurora_deployment
#PBS -o logs/
#PBS -e logs/
#PBS -m abe
#PBS -M avasan@anl.gov

###########################################################################
# Logging
###########################################################################
TAG=$(date +%F_%H%M%S)
OUTPUT_DIR=/dev/shm/ #/lus/flare/projects/Aurora_deployment/avasan/Scaling_Inference/TensorFlow/ST_ZINC22_2D/logs
WORKDIR=/lus/flare/projects/Aurora_deployment/avasan/Scaling_Inference/TensorFlow/Uchic_Aur_Screens/NMNAT_2/Inference_multiconfs
OUTPUTFILE=${TAG}_inf_an_boris_2023.2.log
#PBS -q lustre_test
HOSTNAME=`hostname`
#echo "Running on host: $HOSTNAME"
#export PALS_PING_PERIOD=500
#export PALS_RPC_TIMEOUT=500

cd ${WORKDIR}

###########################################################################
# Setup for wrapper.sh script
###########################################################################
# To ensure GPU affinity mask matches the physical order of the GPUs on the node
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

# The following four env variables are used by wrapper.sh in the 
# mpirun invocation command shown below.
# wrapper.sh is located at: https://gitlab.devtools.intel.com/-/snippets/2387

# These four variables are not needed if you are not using wrapper.sh
export AFFINITY_ORDERING=compact #roundrobin # compact or roundrobin
export RANKS_PER_TILE=4            # processes per tile, change 'x' as needed

export PLATFORM_NUM_GPU=6          # GPUs per node
export PLATFORM_NUM_GPU_TILES=4    # tiles per GPU, use only 1 on PVC A0!

# Set environment variable to print MPICH's process mapping to cores:
export HYDRA_TOPO_DEBUG=1

#############################################
# Load and unload different modules. Activate conda env
#############################################

###########################################################################
# Launch application
###########################################################################

if [ $((RANKS_PER_TILE)) == 1 ]; then
    export ZEX_NUMBER_OF_CCS=0:1,1:1,2:1,3:1,4:1,5:1
    export RANKS_PER_NODE=12
    # CPU affinity for 12 ranks
    export CPU_AFFINITY="verbose,list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:48-55,152-159:56-63,160-167:64-71,168-175:72-79,176-183:80-87,184-191:88-95,192-199"
fi

if [ $((RANKS_PER_TILE)) == 2 ]; then
    export ZEX_NUMBER_OF_CCS=0:2,1:2,2:2,3:2,4:2,5:2
    export RANKS_PER_NODE=24
    # CPU affinity for 24 ranks
    export CPU_AFFINITY="verbose,list:0-3,104-107:4-7,108-111:8-11,112-115:12-15,116-119:16-19,120-123:20-23,124-127:24-27,128-131:28-31,132-135:32-35,136-139:36-39,140-143:40-43,144-147:44-47,148-151:48-51,152-155:52-55,156-159:56-59,160-163:60-63,164-167:64-67,168-171:68-71,172-175:72-75,176-179:76-79,180-183:80-83,184-187:84-87,188-191:88-91,192-195:92-95,196-199"
fi

if [ $((RANKS_PER_TILE)) == 4 ]; then
    export ZEX_NUMBER_OF_CCS=0:4,1:4,2:4,3:4,4:4,5:4
    export RANKS_PER_NODE=48
    # CPU affinity for 48 ranks
    export CPU_AFFINITY="verbose,list:0-1,104-105:2-3,106-107:4-5,108-109:6-7,110-111:8-9,112-113:10-11,114-115:12-13,116-117:14-15,118-119:16-17,120-121:18-19,122-123:20-21,124-125:22-23,126-127:24-25,128-129:26-27,130-131:28-29,132-133:30-31,134-135:32-33,136-137:34-35,138-139:36-37,140-141:38-39,142-143:40-41,144-145:42-43,146-147:44-45,148-149:46-47,150-151:48-49,152-153:50-51,154-155:52-53,156-157:54-55,158-159:56-57,160-161:58-59,162-163:60-61,164-165:62-63,166-167:64-65,168-169:66-67,170-171:68-69,172-173:70-71,174-175:72-73,176-177:74-75,178-179:76-77,180-181:78-79,182-183:80-81,184-185:82-83,186-187:84-85,188-189:86-87,190-191:88-89,192-193:90-91,194-195:92-93,196-197:94-95,198-199"
fi


export PLATFORM_NUM_GPU_TILES=2
export CCL_LOG_LEVEL="DEBUG"

# More CPU affinity per rank
###########################################
#Transfer data to /dev/shm/
###########################################################################

mpiexec -np $(cat $PBS_NODEFILE | wc -l) -ppn 1 --pmi=pmix hostname > SST_Scripts/hostnamelist.dat
export NUM_NODES=8
export TOTAL_NUMBER_OF_RANKS=$((NUM_NODES * RANKS_PER_NODE))
export NUMEXPR_MAX_THREADS=208


module use /soft/modulefiles
module load frameworks/2024.1

aprun --pmi=pmix -n $NUM_NODES -N 1 python /flare/Aurora_deployment/AuroraGPT/cache_soft.py \
	  --src /lus/flare/projects/Aurora_deployment/avasan/envs/sst_2024.tar.gz \
	  --dst /tmp/sst_2024.tar.gz --d

conda activate /tmp/sst_2024
echo "activated conda env"

#mpiexec -np $NUM_NODES -ppn 1 cp -r /lus/flare/projects/Aurora_deployment/avasan/Scaling_Inference/TensorFlow/Uchic_Aur_Screens/NMNAT_2/Training_multiconfs /dev/shm/

#mpiexec -np $NUM_NODES -ppn 1 cp -r /lus/flare/projects/Aurora_deployment/avasan/Scaling_Inference/TensorFlow/Uchic_Aur_Screens/Data/MCU /dev/shm

#mpiexec -np $NUM_NODES -ppn 1 cp -r SST_Scripts/VocabFiles /dev/shm/
# For SPRHBM in Cache-Quad mode using wrapper_hbm_quad.sh
export RANKS_PER_NUMANODE=48 # Flat/Quad with 48 ranks, Flat/Quad has 2 HBM nodes
export MEM_NODE_START_ID=2   # HBM numa ID starts from 2

# ww28 wheels include CCL
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_ALLREDUCE=topo

LOCAL_BATCH_SIZE=1
DATA_FORMAT="channels_last"
PRECISION="float32"
#$BIND_TO

#cd /dev/shm/SST_Scripts/
#echo $PWD
export PWD_flare=/lus/flare/projects/Aurora_deployment/avasan/Scaling_Inference/TensorFlow/Uchic_Aur_Screens/NMNAT_2/Inference_multiconfs
export EXE=/lus/flare/projects/Aurora_deployment/avasan/Scaling_Inference/TensorFlow/Uchic_Aur_Screens/NMNAT_2/Inference_multiconfs/SST_Scripts/run_inf.sh

#############################################
# Load and unload different modules. Activate conda env
#############################################


export ITEX_LIMIT_MEMORY_SIZE_IN_MB=8192
export ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=0

mpiexec -np $TOTAL_NUMBER_OF_RANKS -ppn $RANKS_PER_NODE -hostfile $PBS_NODEFILE --spindle --label --pmi=pmix --cpu-bind ${CPU_AFFINITY}  $PWD_flare/SST_Scripts/set_ze_mask_multiinstance.48ppn.sh ./interposer.sh ${EXE} 2>&1 | tee /dev/shm/${OUTPUTFILE}
mpiexec -np $NUM_NODES -ppn 1 python ./copy_out.py

