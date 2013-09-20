#!/bin/sh
hosts=`cat $1`


for i in `echo $hosts`; do
  host=`echo $i`
  scp -i /home/ubuntu/.ssh/supermario.pem -r /home/ubuntu/sparseallreduce/scala ubuntu@$host:sparseallreduce/
  scp -i /home/ubuntu/.ssh/supermario.pem /home/ubuntu/sparseallreduce/sparsecomm.jar ubuntu@$host:sparseallreduce/sparsecomm.jar

done

scp /home/ubuntu/sparseallreduce/hosts-ip hzhao@stout.cs.berkeley.edu:data/TwitterGraph
