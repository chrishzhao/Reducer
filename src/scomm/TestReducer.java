package scomm;

import mpi.MPIException;
import java.util.*;

public class TestReducer {
	
	public static void main(String [] args) throws MPIException{
		
		Reducer r = new Reducer(args);
		
		int rank = r.rank;
		int size = r.size;
		int modelSize = 160;
		int hostSize = (modelSize + size - 1)/size;
		
		int[] outboundIndices = new int[hostSize*5];
		float[] outboundValues = new float[hostSize*5];
		
		int[] hostIndices = new int[hostSize];
		float[] hostValues = new float[hostSize];
		
		int[] inboundIndices = new int[hostSize*16];
		float[] inboundValues = new float[hostSize*16];

		for(int i = 0; i<hostSize*5; i++){
			
			outboundIndices[i] = (rank*hostSize + i) % modelSize;
			outboundValues[i] =  5 + rank;
		}
		
		for(int i = 0; i<hostSize*16; i++){
			
			inboundIndices[i] = (rank*hostSize + i) % modelSize;
		}

		Arrays.sort(outboundIndices);
		Arrays.sort(inboundIndices);

		r.init();
		if(rank == 0){
			System.out.println("start config ...");
		}		
		r.config(outboundIndices, inboundIndices);
	
		if(rank == 0){
			System.out.println("start reduce ...");
		}
		r.reduce(outboundValues, inboundValues);

		r.terminate();		


		if(rank == 8){
			System.out.println("Inbound Vertices:");
			for(int i = 0; i<inboundValues.length; i++){
				System.out.println(String.format("%d, %f", inboundIndices[i], inboundValues[i]));
			}
		}
	}

}

