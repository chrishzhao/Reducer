package Comm;


import mpi.MPIException;

public class TestReducer {
	
	public static void main(String [] args) throws MPIException{
		
		Reducer r = new Reducer(args);
		
		int rank = r.rank;
		int size = r.size;
		int modelSize = 160;
		int hostSize = (modelSize + size - 1)/size;
		
		int[] outboundIndices = new int[hostSize*2];
		float[] outboundValues = new float[hostSize*2];
		
		int[] hostIndices = new int[hostSize];
		float[] hostValues = new float[hostSize];
		
		for(int i = 0; i<hostSize*2; i++){
			
			outboundIndices[i] = rank*hostSize + i;
			outboundValues[i] = rank;
		}
		
		r.init2d();
		r.scatterConfig2d(outboundIndices, hostIndices);
		r.scatter2d(outboundValues, hostValues);
		
	}

}
