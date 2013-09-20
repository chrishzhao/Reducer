
N=4
TYPE="cc1.4xlarge"


ec2-run-instances ami-e6b7d08f -n $N -g template-all-access -k supermario -t $TYPE --placement-group sparseallreduce --availability-zone us-east-1a


sleep 240


ec2-describe-instances --filter "instance-type=$TYPE" | grep -o 'ip[0-9-]\+' > /home/ubuntu/sparseallreduce/hosts


ec2-describe-instances --filter "instance-type=$TYPE" | grep -o 'ec2-[0-9-]\+' > /home/ubuntu/sparseallreduce/hosts-ip

