# Assignment2_mverhelle

### COMPILATION AND RUN INSTRUCTIONS (From /home/mverhelle directory, for mac terminal):
- Compile A2
`g++ -O2 -fopenmp /home/mverhelle/Assignment2_mverhelle/A2.cpp -o /home/mverhelle/Assignment2_mverhelle/A2`
- Run A2
`./Assignment2_mverhelle/A2`

### INSTRUCTIONS FOR V_TUNE (From /home/mverhelle directory, for mac terminal):
- Run V_Tune GUI
``

### MANUAL INSTRUCTIONS FOR V_TUNE (From /home/mverhelle directory, for mac terminal):
- Load VTune
`module use /software/intel/oneapi_hpc_2022.1/modulefiles`
`module load vtune/latest`
- Run VTune in terminal:
`vtune -collect hotspots -result-dir vtune_result_dir -- /home/mverhelle/Assignment1_mverhelle/A1`
- Run VTune GUI:
`vtune-gui &`