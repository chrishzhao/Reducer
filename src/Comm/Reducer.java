package Comm;

import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;

import mpi.*;

public class Reducer {
	
	// system setup parameters
	public int rank;
	public int size;
	
	// routing table for 1d, 2d case, maybe thinking to change it to hashmap
	private LinkedList<Integer> [] gatherDest;
	private LinkedList<Integer> [] scatterOrigin;
	
	// 2d setup
	private int hSize;
	private int vSize;
	private int hPos;
	private int vPos;
	//private int[] hRange;
	//private int[] vRange;


	// performance measure
	private int byteCount = 0;
	private long nanoTime = 0;
	private long sTime = 0;
	private long eTime = 0;
	
	public Reducer( String[] args ) throws MPIException{
		
		MPI.Init(args);
		rank = MPI.COMM_WORLD.Rank();
		size = MPI.COMM_WORLD.Size();
	}
	
	public void init(){
		gatherDest = (LinkedList<Integer>[]) new LinkedList[size];
		scatterOrigin = (LinkedList<Integer>[]) new LinkedList[size];
	}
	
	public void init2d(){
		gatherDest = (LinkedList<Integer>[]) new LinkedList[size];
		scatterOrigin = (LinkedList<Integer>[]) new LinkedList[size];
		
		//currently only handle size is a square of an integer
		hSize = (int)Math.sqrt((double)size);
		vSize = (int)Math.sqrt((double)size);
		
		//hRange = new int[hSize];
		//vRange = new int[vSize];
		
		//row number
		vPos = rank / hSize;
		
		//column number
		hPos = rank % hSize;
		
		//for(int i = 0; i < hSize; i++){
		//	hRange[i] = h*hSize + i;
		//}
		
		//for(int i = 0; i < vSize; i++){
		//	vRange[i] = v + hSize*i;
		//}
	}
	
	
	public void config2d(int[] outboundIndices, int[] outboundCounts, int[] outboundDispls, int[] inboundIndices, 
						int[] inboundCounts, int[] inboundDispls) throws MPIException {
		//this is wrong!! two directions are different!!
		scatterConfig2d(outboundIndices, outboundCounts, outboundDispls);

	}
	
	public void sparseAllReduce2d(int[] outboundValues, int[] outboundCounts, int[] outboundDispls, int[] inboundValues, 
			int[] inboundCounts, int[] inboundDispls) throws MPIException {
		scatter2d(outboundValues, outboundCounts, outboundDispls);
	}
	
	public int getHostRank(int index){
		return index/size;
	}
	
	public int getRightRank(int i){
		return vPos * hSize + (rank + i) % hSize;
	}
	
	public int getLeftRank(int i){
		return vPos * hSize + (rank + hSize - i) % hSize;
	}
	
	public int getDownRank(int i){
		return (rank + i*hSize) % size;
	}
	
	public int getUpRank(int i){
		return (rank + size - i*hSize) % size;
	}
	
	public int getNeighbourRank(int rank, int j, char orient){
		if(orient == 'h'){
			int hPos = rank % hSize;
			return (hPos + j * hSize);
		}
		
		if(orient == 'v'){
			int vPos = rank / hSize;
			return (vPos * hSize + j);
		}
		
		return -1;
	}
	
	// horizontal scatter map (assuming outboundIndices are sorted according to host rank)
	public void setupScatterMapH(int[] outboundIndices, int[] sendBuffer, int[] sendCounts, int[] sendDispls){
		
		int[] counts = new int[size];
		int[] displs = new int[size+1];
		
		int lRank = -1;
		int cRank = 0;

		for( int i = 0; i < outboundIndices.length; i++ ){
			cRank = getHostRank( outboundIndices[i] );
			if( lRank != cRank ){
				for( int j = lRank+1; j <= cRank; j++ ){
					displs[j] = i;
				}
			}
			lRank = cRank;
		}
		
		for( int i = lRank+1; i <= size; i++){
			displs[i] = outboundIndices.length;
		}
		
		for( int i = 0; i < size; i++){
			counts[i] = displs[i+1] - displs[i];
		}
		
		// add carry-over indices to the table
		
	}
	
	// vertical scatter map (after horzontal scatter)
	public void setupScatterMapV(){
		
	}
	
   
	// the basic scatter config element
	public void scatterConfig(int[] sendBuffer, int[] sendCounts, int[] sendDispls, char orient) throws MPIException{
		
		int [] recvCounts = new int[size];
		int [] recvBuffer;
		
		for(int i = 0; i < hSize; i++){
			
			int right =  (orient == 'h') ? getRightRank(i) : getDownRank(i);
			int left = (orient == 'h') ? getLeftRank(i) : getUpRank(i);
			
			MPI.COMM_WORLD.Sendrecv(sendCounts, i, 1, MPI.INT, right, 0, recvCounts, left, 1, MPI.INT, left, 0);
				
			recvBuffer = new int[recvCounts[left]];
			
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.INT, right, 0, recvBuffer, 0, recvCounts[left], MPI.INT, left, 0);
		
			scatterOrigin[left] = new LinkedList<Integer>();
			for(int j = 0; j<recvCounts[left]; j++){
				scatterOrigin[left].add(recvBuffer[j]);
			}
		}
		
	}
	
	public void scatterConfig2d(int[] outboundIndices) throws MPIException{
		
		int[] sendBufferH = new int[outboundIndices.length];
		int[] sendCountsH = new int[hSize];
		int[] sendDisplsH = new int[hSize + 1];
		
		setupScatterMapH(outboundIndices, sendBufferH, sendCountsH, sendDisplsH);
		scatterConfig(sendBufferH, sendCountsH, sendDisplsH, 'h');
		
		
		int[] sendBufferV = new int[outboundIndices.length];
		int[] sendCountsV = new int[vSize];
		int[] sendDisplsV = new int[vSize + 1];
		setupScatterMapV(outboundIndices, sendBufferV, sendCountsV, sendDisplsV);
		scatterConfig(sendBufferV, sendCountsV, sendDisplsV, 'v');
	}
	
	public void scatterConfig2d(int[] outboundIndices, int[] outboundCounts, int[] outboundDispls) throws MPIException{
		
		int [] sendCounts = new int[size];
		int [] recvCounts = new int[size];
		int [] sendBuffer;
		int [] recvBuffer;

		//int vPosRight = right / hSize;
		//int hPosRight = right % hSize;
		
		
		for(int i = 1; i < hSize; i++){
			
			int right =  getRightRank(i);
			int left = getLeftRank(i);
			
			sendCounts[right] = 0;
					
			for(int j = 0; j<vSize; j++){
				int nRank = getNeighbourRank(right, j, 'h');
				sendCounts[right] += outboundCounts[nRank];
			}
		
			MPI.COMM_WORLD.Sendrecv(sendCounts, right, 1, MPI.INT, right, 0, recvCounts, left, 1, MPI.INT, left, 0);
		
			sendBuffer = new int[sendCounts[right]];
			int sendBufferPointer = 0;
					
			for(int j = 0; j<vSize; j++){
				int nRank = getNeighbourRank(right, j, 'h');
				System.arraycopy(outboundIndices, outboundDispls[nRank], sendBuffer, sendBufferPointer, outboundCounts[nRank]);
				sendBufferPointer += outboundCounts[nRank];
			}
		
			recvBuffer = new int[recvCounts[left]];
			MPI.COMM_WORLD.Sendrecv(sendBuffer, 0, sendCounts[right], MPI.INT, right, 0, recvBuffer, 0, recvCounts[left], MPI.INT, left, 0);
		
			scatterOrigin[left] = new LinkedList<Integer>();
			for(int j = 0; j<recvCounts[left]; j++){
				scatterOrigin[left].add(recvBuffer[j]);
			}
		}
		
		for(int i = 1; i < vSize; i++){
			
			int down = getDownRank(i);
			int up = getUpRank(i);
			
			sendCounts[down] = 0;
			
			
		}
	}
	
	
	public void scatter2d(int[] outboundValues, int[] outboundCounts, int[] outboundDispls){
		
	}
	
	
	public void gatherConfig2d(int[] inboundIndices, int[] inbound_counts, int[] inbound_displs){
		
	}
	
	public void gather2dh(){
		
	}
	
	public void gather2dv(){
		
	}
	

}
