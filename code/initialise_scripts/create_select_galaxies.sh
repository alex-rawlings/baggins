#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    echo "No galaxy name provided!"
    echo "usage:"
    echo "    ./create_select_galaxies.sh path name1 name2 ... nameN"
    echo "path: path to parameter file directory"
    echo "name_: parameter file name"
    exit 1
fi

#create initial conditions for the specified galaxies
param_dir=$1
#add a slash if not in the path
if [[ ${param_dir: -1} != "/" ]]
then
    param_dir="$param_dir/"
fi

#run the pipeline
echo "------------------------------------"
echo "Running the initialisation pipeline"
echo "------------------------------------"

first_arg_flag=1
file_ext=""
for galaxy in $@
do
    if [ $first_arg_flag -eq 1 ]
    then
        first_arg_flag=0
        continue
    elif [[ ${galaxy: -3} != ".py" ]]
    then
        file_ext=".py"
    fi
    sim="$param_dir$galaxy$file_ext"
    echo "Running: $sim"
    python ./bh_dm_mass.py  $sim
    python ./galaxy_gen.py $sim -u -v
    python ./ic_kinematics.py $sim
    python ./view_gal.py $sim -v --numrotations 2
    echo $'\n'
done
