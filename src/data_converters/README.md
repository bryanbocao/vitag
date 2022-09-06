# Data Converters

To run the code, make sure the dataset ```RAN``` is available with the following structure:
```
RAN
 |-seqs
    |-indoor
       |-scene0
         |-20201223_140951
            |-Depth
            |-GND
            |-IMU
            |-IMU_NED_pos
            |-RGB_ts16_dfv4_anonymized
            |-WiFi
         |-...(x14)
    |-outdoor
       |-scene1
         |-20211004_142306
            |-Depth
            |-GND
            |-IMU
            |-IMU_NED_pos
            |-RGB_ts16_dfv4_anonymized
            |-WiFi
         |-...(x15)
```
where ```scene0``` contains 15 sequences and ```scene1``` consists of 16 sequences, respectively. Due to the privacy issues mentioned in our ```IRB```, we only provide the processed dataset ```RAN4model```. The script of ```data_converters``` are for reference.
