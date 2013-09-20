import java.util.Arrays;
import sparsecomm.*;
import mpi.*;

class Model {

	public int[] inboundIndices;
	public float[] inboundValues;
	public int[] outboundIndices;
	public float[] outboundValues;
	
	public float[] vector;
	
	// dimension of model
	public int dim;
	
	// dimension per processoer
	public int dim_per_proc;

	public SparseComm comm;
	// number of processors
	public int size;
	
	public int[] inbound_displs;
	public int[] inbound_counts;
	public int[] outbound_displs;
	public int[] outbound_counts;
	
	// command line arguments
	public String[] argv;
	
	public Model(String[] args){
		argv = args;
		dim = Integer.parseInt(argv[0]);
	}
	
	public void initModel() throws MPIException{
		
		int sparsity = 6;
		inboundIndices = new int[sparsity];
		inboundValues = new float[sparsity];
		outboundIndices = new int[sparsity];
		outboundValues = new float[sparsity];
		
		comm = new SparseComm(argv);
		size = comm.size;
		
		// dummy model for now
		for( int i = 0; i<sparsity; i++ ){
			inboundIndices[i] = ((comm.rank*3 + 2 + i)%dim);
			inboundValues[i] = 7 + comm.rank*i;
			outboundIndices[i] = ((comm.rank*3 + 1 + i)%dim);
			outboundValues[i] = 9 + comm.rank*i*2;
		}
		
		Arrays.sort(inboundIndices);
		Arrays.sort(outboundIndices);
		
		dim_per_proc = (dim + size -1)/size;
		vector = new float[dim_per_proc];
		for(int i=0; i<dim_per_proc; i++){
			vector[i] = 0;
		}
		
		//for(int i = 0; i<sparsity; i++){
		//	if(comm.rank == outboundIndices[i]/dim_per_proc){
		//		int k = outboundIndices[i] % dim_per_proc;
		//		vector[k] = outboundValues[i];
		//	}
		//}
		
		// partition
		partition();
		
		System.out.println(String.format("Processor %d inbound: %s %s\n", comm.rank, Arrays.toString(inboundIndices), Arrays.toString(inboundValues)));
		System.out.println(String.format("Processor %d outbound: %s %s\n", comm.rank, Arrays.toString(outboundIndices), Arrays.toString(outboundValues)));

		//System.out.println(String.format("Processor %d inbound displs: %d, %d, %d, %d, %d\n", comm.rank, inbound_displs[0], inbound_displs[1], inbound_displs[2], inbound_displs[3], inbound_displs[4]));
		//System.out.println(String.format("Processor %d outbound displs: %d, %d, %d, %d, %d\n", comm.rank, outbound_displs[0], outbound_displs[1], outbound_displs[2], outbound_displs[3], outbound_displs[4]));
		

	}
	
	public void partition(){
			
		//int dim_per_proc = (dim + size -1)/size;
		inbound_displs = new int[size+1];
		inbound_counts = new int[size];
		outbound_displs = new int[size+1];
		outbound_counts = new int[size];
		
		int lpid = -1;
		int pid = 0;

		for(int i = 0; i < inboundIndices.length; i++){
			pid = inboundIndices[i]/dim_per_proc;
			if( lpid != pid ){
				for(int j=lpid+1; j<=pid; j++){
					inbound_displs[j] = i;
				}
			}
			lpid = pid;
		}
		
		for(int i = lpid+1; i<=size; i++){
			inbound_displs[i] = inboundIndices.length;
		}
		
		for( int i = 0; i < size; i++ ){
			inbound_counts[i] = inbound_displs[i+1] - inbound_displs[i];
		}
		
		
		lpid = -1;
		pid = 0;

		for(int i = 0; i < outboundIndices.length; i++){
			pid = outboundIndices[i]/dim_per_proc;
			if( lpid != pid ){
				for(int j=lpid+1; j<=pid; j++){
					outbound_displs[j] = i;
				}
			}
			lpid = pid;
		}
		
		for(int i = lpid+1; i<=size; i++){
			outbound_displs[i] = outboundIndices.length;
		}
		
		for( int i = 0; i < size; i++ ){
			outbound_counts[i] = outbound_displs[i+1] - outbound_displs[i];
		}
	
	}
	
	//public void reduce() throws MPIException{
	//	comm.DenseAllReduce(model, partition_offsets, partition_sizes);
	//}
	
	public void reduce() throws MPIException{
		comm.scatterConfig(outboundIndices, outbound_counts, outbound_displs);
		comm.gatherConfig(inboundIndices, inbound_counts, inbound_displs);
		
		
		//if(comm.rank==2){
		//	comm.printConfig();
		//}
		
		comm.scatter(outboundValues, outbound_counts, outbound_displs, vector,  dim_per_proc);
		System.out.println(String.format("Processor %d vector after scatter: %s\n", comm.rank, Arrays.toString(vector)));
		comm.gather(vector, inboundValues, inbound_counts, dim_per_proc);
		System.out.println(String.format("Processor %d inbound after gather: %s\n", comm.rank, Arrays.toString(inboundValues)));

	}
	
	// update the model
	public void update(){
	
	}

	public void run() throws MPIException{
	
		update();

		reduce();

		
		//System.out.println(String.format("Processor %d inbound: (%d, %f) (%d, %f) (%d, %f)\n", comm.rank, inboundIndices[0], inboundValues[0], inboundIndices[1], inboundValues[1], inboundIndices[2], inboundValues[2]));
		//System.out.println(String.format("Processor %d outbound: (%d, %f) (%d, %f) (%d, %f)\n", comm.rank, outboundIndices[0], outboundValues[0], outboundIndices[1], outboundValues[1], outboundIndices[2], outboundValues[2]));

	
	}
	public void terminate() throws MPIException{
		comm.terminate();
	}
	
	
    static public void main(String[] args) throws MPIException{
		
		Model m = new Model(args);
		
		m.initModel();
		m.run();
		m.terminate();
      
    }
}
