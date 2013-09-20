#!/bin/bash

#scalac -cp $ALL_LIBS:/usr/local/lib/mpi.jar:/home/ubuntu/sparseallreduce/sparsecomm.jar Model.scala

#mpirun -np 2 java -Xmx28G -Djava.library.path=$LD_LIBRARY_PATH -cp $ALL_LIBS:/home/ubuntu/lib/BIDMat/lib/jhdf5.jar:/home/ubuntu/sparseallreduce/sparsecomm.jar:/home/ubuntu/lib/scala-2.9.2/lib/scala-library.jar:/home/ubuntu/lib/scala-2.9.2/lib/jline.jar:. Model 41652230

mpirun -x LD_LIBRARY_PATH -np 32 -nolocal -hostfile /home/ubuntu/sparseallreduce/hosts java -Xmx28G -Djava.library.path=$LD_LIBRARY_PATH -cp $ALL_LIBS:/home/ubuntu/lib/BIDMat/lib/jhdf5.jar:/home/ubuntu/sparseallreduce/sparsecomm.jar:/home/ubuntu/lib/scala-2.9.2/lib/scala-library.jar:/home/ubuntu/lib/scala-2.9.2/lib/jline.jar:. Model 41652230 > output-32-1.txt
mpirun -x LD_LIBRARY_PATH -np 32 -nolocal -hostfile /home/ubuntu/sparseallreduce/hosts java -Xmx28G -Djava.library.path=$LD_LIBRARY_PATH -cp $ALL_LIBS:/home/ubuntu/lib/BIDMat/lib/jhdf5.jar:/home/ubuntu/sparseallreduce/sparsecomm.jar:/home/ubuntu/lib/scala-2.9.2/lib/scala-library.jar:/home/ubuntu/lib/scala-2.9.2/lib/jline.jar:. Model 41652230 > output-32-2.txt
mpirun -x LD_LIBRARY_PATH -np 32 -nolocal -hostfile /home/ubuntu/sparseallreduce/hosts java -Xmx28G -Djava.library.path=$LD_LIBRARY_PATH -cp $ALL_LIBS:/home/ubuntu/lib/BIDMat/lib/jhdf5.jar:/home/ubuntu/sparseallreduce/sparsecomm.jar:/home/ubuntu/lib/scala-2.9.2/lib/scala-library.jar:/home/ubuntu/lib/scala-2.9.2/lib/jline.jar:. Model 41652230 > output-32-3.txt
mpirun -x LD_LIBRARY_PATH -np 16 -nolocal -hostfile /home/ubuntu/sparseallreduce/hosts java -Xmx28G -Djava.library.path=$LD_LIBRARY_PATH -cp $ALL_LIBS:/home/ubuntu/lib/BIDMat/lib/jhdf5.jar:/home/ubuntu/sparseallreduce/sparsecomm.jar:/home/ubuntu/lib/scala-2.9.2/lib/scala-library.jar:/home/ubuntu/lib/scala-2.9.2/lib/jline.jar:. Model 41652230 > output-16-1.txt
mpirun -x LD_LIBRARY_PATH -np 16 -nolocal -hostfile /home/ubuntu/sparseallreduce/hosts java -Xmx28G -Djava.library.path=$LD_LIBRARY_PATH -cp $ALL_LIBS:/home/ubuntu/lib/BIDMat/lib/jhdf5.jar:/home/ubuntu/sparseallreduce/sparsecomm.jar:/home/ubuntu/lib/scala-2.9.2/lib/scala-library.jar:/home/ubuntu/lib/scala-2.9.2/lib/jline.jar:. Model 41652230 > output-16-2.txt
mpirun -x LD_LIBRARY_PATH -np 16 -nolocal -hostfile /home/ubuntu/sparseallreduce/hosts java -Xmx28G -Djava.library.path=$LD_LIBRARY_PATH -cp $ALL_LIBS:/home/ubuntu/lib/BIDMat/lib/jhdf5.jar:/home/ubuntu/sparseallreduce/sparsecomm.jar:/home/ubuntu/lib/scala-2.9.2/lib/scala-library.jar:/home/ubuntu/lib/scala-2.9.2/lib/jline.jar:. Model 41652230 > output-16-3.txt





