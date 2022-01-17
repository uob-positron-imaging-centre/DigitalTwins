set -e
module purge; module load bluebear


# Contains pept and OpenCV
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
module load PICI-LIGGGHTS/20210202-foss-2020a-Python-3.8.2
module load sympy/1.6.2-foss-2020a-Python-3.8.2

# Create virtual environment for installing Coexist
export VENV_DIR="${HOME}/virtual-environments"
export VENV_PATH="${VENV_DIR}/coexist-${BB_CPU}"

# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
    echo "Virtual environment did not exist. Installing it now would create a race condition. Install coexist for all architectures first."
    exit 1
else
    source ${VENV_PATH}/bin/activate
fi


export PMIX_MCA_gds=hash

# Print the command line arguments to the terminal
echo $*
