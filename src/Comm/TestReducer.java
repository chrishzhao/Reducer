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
		
		int[] outboundIndices = new int[hostSize*16];
		float[] outboundValues = new float[hostSize*16];
		
		int[] hostIndices = new int[hostSize];
		float[] hostValues = new float[hostSize];
		
		for(int i = 0; i<hostSize*16; i++){
			
			outboundIndices[i] = (rank*hostSize + i) % modelSize;
			outboundValues[i] =  5 + rank;
		}
		
		Arrays.sort(outboundIndices);
		r.init2d();
		if(rank == 0){
			System.out.println("start config ...");
		}		
		r.scatterConfig2d(outboundIndices, hostIndices);
		if(rank == 0){
			System.out.println("start scatter ...");
		}
		r.scatter2d(outboundValues, hostValues);
		r.terminate();		


		if(rank == 7){
			System.out.println("Indices");
			for(int i = 0; i<hostSize; i++){
			
				System.out.println(String.format("%d, %f", hostIndices[i], hostValues[i]));
			}
		}
	}

}
