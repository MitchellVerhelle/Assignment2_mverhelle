# Assignment2_mverhelle

### COMPILATION AND RUN INSTRUCTIONS (From /home/mverhelle directory, for mac terminal):
- c_cpp_properties.json (Not necessary to run, just convenient for editor like vscode) to add to include path manually
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/software/eigen-3.4-el8-x86_64/include/eigen3"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "gnu++14",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
#### SETUP ENV
- EITHER RUN THIS:
`/home/mverhelle/Assignment2_mverhelle/setup_env.sh`
- OR SETUP ENV MANUALLY WITH:
`module load intel/2022.0`
`module load eigen/3.4`
`module load vtune/latest`
`module use /software/intel/oneapi_hpc_2022.1/modulefiles`

#### **sinteractive, compile, and run**
- Request however many cores
`sinteractive --time=0:05:0 --cpus-per-task=8 --account=finm32950`
- Compile A2
`icc -std=c++17 -qopenmp -I$EIGEN3_DIR -o /home/mverhelle/Assignment2_mverhelle/A2 /home/mverhelle/Assignment2_mverhelle/A2.cpp`
- - If this doesn't run, manually run top two env module loads in terminal. Then it should work.
- Run A2
`/home/mverhelle/Assignment2_mverhelle/A2`

### MANUAL INSTRUCTIONS FOR V_TUNE (From /home/mverhelle directory, for mac terminal):
- Run VTune in terminal: (Delete vtune_result folder to rerun)
`vtune -collect hotspots -result-dir /home/mverhelle/Assignment2_mverhelle/vtune_result -- /home/mverhelle/Assignment2_mverhelle/A2`
- Run VTune GUI:
`vtune-gui &`