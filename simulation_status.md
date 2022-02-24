# Simulation Status  

## Overview  
Legend:  

|Key|Description      |Total|
|---|-----------------|-----|
|h  | hold (resources)|   0 |
|w  | waiting p --> c |   0 |
|p  | parent running  |   0 |
|c  | children running|  12 |
|C  | set completed   |  32 |
|-  | abandoned       |  24 |


|     |AB|AC|AD|AE|BC|BD|BE|CD|CE|DE|
|-----|--|--|--|--|--|--|--|--|--|--|
|0.001|- |C |C |C |- |- |- |c |C |C |
|0.005|- |C |C |C |- |- |- |c |C |C |
|0.010|- |C |C |C |- |- |- |c |C |C |
|0.050|- |C |C |C |- |- |- |c |C |C |
|0.100|- |C |C |C |- |- |- |c |C |C |
|1.000|- |C |C |C |- |- |- |C |C |C |


## Individual Runs  
Legend:  

|Key|Description    |Total|
|---|---------------|-----|
|.  | running       |  20 |
|o  | completed     | 103 |
|x  | abandoned     | 226 |
|?  | unsure        |   0 |
|>  |>13.8Gyr-Merger|  11 |


|Configuration|000|001|002|003|004|005|006|007|008|009|Total|
|-------------|---|---|---|---|---|---|---|---|---|---|-----|
|AC-3.0-0.001 | x | x | x | x | o | o | x | x | x | x |   2 |
|AC-3.0-0.005 | x | o | o | o | x | o | x | o | x | o |   6 |
|AC-3.0-0.010 | x | x | x | x | o | o | o | x | x | o |   4 |
|AC-3.0-0.050 | x | o | o | x | o | x | x | o | x | o |   5 |
|AC-3.0-0.100 | o | x | x | o | o | o | o | x | o | o |   7 |
|AC-3.0-1.000 | x | x | x | x | x | x | o | x | x | o |   1 |
|             |   |   |   |   |   |   |   |   |   |   |     |
|AD-3.0-0.001 | x | x | x | x | x | x | x | x>| o | x |   1 |
|AD-3.0-0.005 | x | o | x | x | x | x | x | x | x | x |   1 |
|AD-3.0-0.010 | x | x | x | x | x | x | x | x>| o | o |   2 |
|AD-3.0-0.050 | o | x | x | x | x | x | o | x | x | x>|   2 |
|AD-3.0-0.100 | x | x | x | o | x | x | o | x | o | x |   3 |
|AD-3.0-1.000 | o | o | x>| o | o | o | x | x | o | x |   6 |
|             |   |   |   |   |   |   |   |   |   |   |     |
|AE-3.0-0.001 | x | x | x | x | x | x | x | x | x | x |   0 |
|AE-3.0-0.005 | x | o | x | x>| x>| x | x | x | x | x |   1 |
|AE-3.0-0.010 | x | x | x | x | x | x | o | x | x | x |   1 |
|AE-3.0-0.050 | o | x | x>| x | x | x | x | x | x | x |   1 |
|AE-3.0-0.100 | x | x | x | x | x | x | x | x | x | x |   0 |
|AE-3.0-1.000 | x | x | > | x | x | o | x | o | x | x |   2 |
|             |   |   |   |   |   |   |   |   |   |   |     |
|CD-3.0-0.001 | x | x | . | x | x | . | . | x | x | x |     |
|CD-3.0-0.005 | . | o | . | x | . | . | o | x | x | . |   2 |
|CD-3.0-0.010 | o | . | . | x | x | x | x | x | . | . |   1 |
|CD-3.0-0.050 | . | . | x | x | . | x | x | x | x | x |     |
|CD-3.0-0.100 | x | x | . | . | . | o | x | x | . | . |   1 |
|CD-3.0-1.000 | x | x | x | o | o | o | o>| o | o | o |   6 |
|             |   |   |   |   |   |   |   |   |   |   |     |
|CE-3.0-0.001 | x | x | x | x | x | o | x | x | x | x |   1 |
|CE-3.0-0.005 | x | o | x | o | x | x | o | o | o | x |   5 |
|CE-3.0-0.010 | x | x | x | x | x | x | x | x | x | x |   0 |
|CE-3.0-0.050 | x | x | o | o | x>| o | o | o | x | x |   5 |
|CE-3.0-0.100 | x | x | x | x | x | x | x | o | x | x |   1 |
|CE-3.0-1.000 | x | x | x | x | x | o | x | x | x | o |   2 |
|             |   |   |   |   |   |   |   |   |   |   |     |
|DE-3.0-0.001 | o | o | x | o | x | o | x | x | o | x |   5 |
|DE-3.0-0.005 | x | x | x | x | x | o | x | x | > | x |   1 |
|DE-3.0-0.010 | x | x | o | o | o | o | o | o | o | x |   7 |
|DE-3.0-0.050 | x | x | x | o | o | o | o | x | o | o |   6 |
|DE-3.0-0.100 | o | o | o | o | o | o | o | o | o | x |   9 |
|DE-3.0-1.000 | x | x | x | o | o | o | o | o | x | x |   5 |
