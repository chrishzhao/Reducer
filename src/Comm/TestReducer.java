package Comm;


import mpi.MPIException;
import java.util.*;

public class TestReducer {
	
	public static void main(String [] args) throws MPIException{
		
		Reducer r = new Reducer(args);
		
		int rank = r.rank;
		int size = r.size;
		int modelSize = 160;
		int hostSize = (modelSize + size - 1)/size;
		
		int[] outboundIndices = new int[hostSize*4];
		float[] outboundValues = new float[hostSize*4];
		
		int[] hostIndices = new int[hostSize];
		float[] hostValues = new float[hostSize];
		
		for(int i = 0; i<hostSize*4; i++){
			
			outboundIndices[i] = (rank*hostSize + i) % modelSize;
			outboundValues[i] =  5 + rank;
		}
		
		Arrays.sort(outboundIndices);
		r.init2d();
		r.scatterConfig2d(outboundIndices, hostIndices);
		r.scatter2d(outboundValues, hostValues);
		r.terminate();		


		if(rank == 12){
			System.out.println("Indices");
			for(int i = 0; i<hostSize; i++){
			
				System.out.println(String.format("%d, %f", hostIndices[i], hostValues[i]));
			}
		}
	}

}
