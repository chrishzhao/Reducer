package scomm;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import mpi.MPI;
import mpi.MPIException;

public class Reducer {
	
	public int rank;
	public int size;
	
	// size of model
	int modelSize;
	
	// k-ary d-fly butterfly network
	int k;
	int d;
	
	// keep track of scatter/gather vertices, of dimension k*d
	private LinkedList<Integer> [] scatterDest;
	private LinkedList<Integer> [] scatterOrigin;
	private LinkedList<Integer> [] gatherDest;
	private LinkedList<Integer> [] gatherOrigin;	
	
	// internal buffers
	private Map<Integer, Float> scatterBuffer;
	private Map<Integer, Float> gatherBuffer;
	private Map<Integer, Float> hostBuffer;
	
	// vid -> index map
	private Map<Integer, Integer> outboundMap;
	private Map<Integer, Integer> inboundMap;
	
	
	public Reducer( String[] args ) throws MPIException{
		
		MPI.Init(args);
		rank = MPI.COMM_WORLD.Rank();
		size = MPI.COMM_WORLD.Size();
		modelSize = Integer.parseInt(args[0]);
		k = Integer.parseInt(args[1]);
		d = Integer.parseInt(args[2]);
	}
	
	
	// return the set of vertex indices (in an array) the current machine host
	public void setHostBuffer(){
		int hostSize = (modelSize+size-1)/size;
		for(int i = 0; i < hostSize; i++){
			hostBuffer.put(rank*hostSize + i, 0f);
		}
	}
		
	public void init(){
		
		scatterDest = (LinkedList<Integer>[]) new LinkedList[d * k];
		scatterOrigin = (LinkedList<Integer>[]) new LinkedList[d * k];
		gatherDest = (LinkedList<Integer>[]) new LinkedList[d * k];
		gatherOrigin = (LinkedList<Integer>[]) new LinkedList[d * k];
		
		scatterBuffer = new HashMap<Integer, Float>();
		gatherBuffer = new HashMap<Integer, Float>();

		outboundMap = new HashMap<Integer, Integer>();
		inboundMap = new HashMap<Integer, Integer>();

		hostBuffer = new HashMap<Integer, Float>();
		setHostBuffer();
	}
	
	public int getRight( int i, int level){
		int intv = (int) Math.pow(k, level);
		int pos = (rank + intv*i) % (k * intv);
		int init = ((rank) / (k * intv))*(k * intv);
		return init + pos;
	}
	
	public int getLeft( int i, int level){
		return getRight( k - i, level);
	}
	
	public int getHost( int vid ){
		int hostSize = (modelSize+size-1)/size;
		return vid/hostSize;
	}
	
	public int getScatterDest( int host, int level){
		
		for( int dest = 0; dest < k; dest ++){
			int right = getRight(dest, level);
			if((right % Math.pow(k, level+1)) == (host % Math.pow(k, level+1))){
				return dest;
			}
		}
		return -1;
	}
	
	public int getGatherDest( int host, int level){
		
		for( int dest = 0; dest < k; dest ++){
			int right = getRight(dest, level);
			if((right / (int)Math.pow(k, level)) == (host / (int)Math.pow(k, level))){
				return dest;
			}
		}
		return -1;
	}
	
	public void makeScatterSendBuffer( float[] sendBuffer, int[] sendCounts, int[] sendDispls, int level){
		
		int bufferPointer = 0;
		
		for(int i = 0; i < k; i++){
			sendCounts[i] = scatterDest[level * k + i].size();
			for(int vid: scatterDest[level * k + i]){
				sendBuffer[bufferPointer++] = scatterBuffer.get(vid);
			}
		}
		
		for( int i = 0; i <= k; i++){
			 for(int t = 0; t < i; t++){
				 sendDispls[i] += sendCounts[t];
			 }
		}
		
	}
	
	public void makeGatherSendBuffer( float[] sendBuffer, int[] sendCounts, int[] sendDispls, int level){
		
		int bufferPointer = 0;
		
		for(int i = 0; i < k; i++){
			sendCounts[i] = gatherDest[level * k + i].size();
			for(int vid: gatherDest[level * k + i]){
				sendBuffer[bufferPointer++] = gatherBuffer.get(vid);
			}
		}
		
		for( int i = 0; i <= k; i++){
			 for(int t = 0; t < i; t++){
				 sendDispls[i] += sendCounts[t];
			 }
		}
		
		
	}
	
	public void makeScatterConfigSendBuffer( int[] sendBuffer, int[] sendCounts, int[] sendDispls, int level){
		
		int[] outboundIndices = new int[sendBuffer.length];
		
		int j = 0;
		for( Integer vid : scatterBuffer.keySet()){
			outboundIndices[j++] = vid;
		}
		
		// sorting according to host rank
		Arrays.sort(outboundIndices);

		for( int i = 0; i < outboundIndices.length; i++ ){
			int host = getHost(outboundIndices[i]);
			int dest = getScatterDest(host, level);
			//System.out.println(String.format("rank: %d, host: %d, level: %d dest: %d", rank, host, level, dest));
			if(dest >= 0){sendCounts[dest] += 1;}
		}
		
		for( int i = 0; i <= k; i++){
			 for(int t = 0; t < i; t++){
				 sendDispls[i] += sendCounts[t];
			 }
		}
		
		int[] pointers = new int[k];
		
		for( int i = 0; i < outboundIndices.length; i++ ){
			int host = getHost(outboundIndices[i]);
			int dest = getScatterDest(host, level);
			if(dest >=0 ){
			sendBuffer[sendDispls[dest] + pointers[dest]] = outboundIndices[i]; 			   
			pointers[dest] += 1;
			}
		}
		
	}
	
	public void makeGatherConfigSendBuffer( int[] sendBuffer, int[] sendCounts, int[] sendDispls, int level){
		
		int[] inboundIndices = new int[sendBuffer.length];
		
		int j = 0;
		for( Integer vid : gatherBuffer.keySet()){
			inboundIndices[j++] = vid;
		}
		
		// sorting according to host rank
		Arrays.sort(inboundIndices);

		for( int i = 0; i < inboundIndices.length; i++ ){
			int host = getHost(inboundIndices[i]);
			int dest = getGatherDest(host, level);
			if(dest >= 0){sendCounts[dest] += 1;}
		}
		
		for( int i = 0; i <= k; i++){
			 for(int t = 0; t < i; t++){
				 sendDispls[i] += sendCounts[t];
			 }
		}
		
		int[] pointers = new int[k];
		
		for( int i = 0; i < inboundIndices.length; i++ ){
			int host = getHost(inboundIndices[i]);
			int dest = getGatherDest(host, level);
			if(dest >= 0){
			sendBuffer[sendDispls[dest] + pointers[dest]] = inboundIndices[i]; 
			pointers[dest] += 1;
			}
		}
	}
	
	public void config( int[] outboundIndices, int[] inboundIndices) throws MPIException{
		
		scatterBuffer.clear();
		gatherBuffer.clear();
		
		int j = 0;
		for( int vid : outboundIndices){
			scatterBuffer.put( vid, 0f );
			outboundMap.put(vid, j++);
		}
		
		j = 0;
		for( int vid : inboundIndices){
			gatherBuffer.put( vid, 0f );
			inboundMap.put(vid, j++);
		}
		
		int [] sendBuffer;
		int [] sendCounts;
		int [] sendDispls;
		
		//System.out.println("scatter config");
		for( int l = 0; l < d; l++){
			
		//	if(rank == 10)System.out.println(String.format("rank: %d, l: %d", rank, l));
			sendBuffer = new int[scatterBuffer.size()];
			sendCounts = new int[k];
			sendDispls = new int[k + 1];
			
			makeScatterConfigSendBuffer( sendBuffer, sendCounts, sendDispls, l);
		
		//	if(rank == 10)System.out.println(String.format("cp 1 rank: %d, l: %d", rank, l));
			scatterConfig( sendBuffer, sendCounts, sendDispls, l);
			
		//	if(rank == 10)System.out.println(String.format("cp 2 rank: %d, l: %d", rank, l));
			for( int i = 0; i < k; i++){
				for( int vid: scatterOrigin[ k * l + i]){
					scatterBuffer.put( vid, 0f );
				}
			}
			
		}
		
		//System.out.println("gather config");
		for( int l = d-1; l>=0; l--){
			//if(rank == 10)System.out.println(String.format("rank: %d, l: %d", rank, l));
			sendBuffer = new int[gatherBuffer.size()];
			sendCounts = new int[k];
			sendDispls = new int[k + 1];
			
			makeGatherConfigSendBuffer( sendBuffer, sendCounts, sendDispls, l);
			//if(rank == 10){
			//System.out.println(String.format("sendbuffer level:%d", l));
			//for(int vid: sendBuffer){System.out.print(String.format("%d ", vid));}
			//System.out.println(String.format("sendcount level:%d", l));
			//for(int vid: sendCounts){System.out.print(String.format("%d ", vid));}
			//System.out.println(String.format("senddispls level:%d", l));
			//for(int vid: sendDispls){System.out.print(String.format("%d ", vid));}
			//}
			gatherConfig( sendBuffer, sendCounts, sendDispls, l);
			
			for( int i = 0; i < k; i++){
				for( int vid: gatherDest[ k * l + i]){
					gatherBuffer.put( vid, 0f );
				}
			}
		}
		
		
	}
	
	public void reduce( float[] outboundValues, float[] inboundValues) throws MPIException{
		
		// reset scatter buffer
		for( int vid: scatterBuffer.keySet() ){
			if(outboundMap.containsKey(vid)){
				scatterBuffer.put(vid, outboundValues[outboundMap.get(vid)]);
			}else{
				scatterBuffer.put(vid, 0f);
			}
		}
		
		// scatter
		float [] sendBuffer;
		int [] sendCounts;
		int [] sendDispls;
		
		for( int l = 0; l<d; l++){
			
			int bufferSize = 0;
			for(int i = 0; i < k; i++){
				bufferSize += scatterDest[l * k + i].size();
			}
			sendBuffer = new float[bufferSize];
			sendCounts = new int[k];
			sendDispls = new int[k + 1];
			
			makeScatterSendBuffer( sendBuffer, sendCounts, sendDispls, l);
			scatter( sendBuffer, sendCounts, sendDispls, l);
		}
		
		// set host buffer
		for( int vid: hostBuffer.keySet()){
			if(scatterBuffer.containsKey(vid)){
				hostBuffer.put(vid, scatterBuffer.get(vid));
			}
		}
		
		// reset gather buffer
		for( int vid: gatherBuffer.keySet()){
			if(hostBuffer.containsKey(vid)){
				gatherBuffer.put(vid, hostBuffer.get(vid));
				//{System.out.println(String.format("rank: %d, vid: %d, val: %f", rank, vid, hostBuffer.get(vid)));}
			}else{
				gatherBuffer.put(vid, 0f);
			}
		}
		
		
		// gather
		for( int l = 0; l<d; l++){
			
			int bufferSize = 0;
			for(int i = 0; i < k; i++){
				bufferSize += gatherDest[l * k + i].size();
			}
			
			sendBuffer = new float[bufferSize];
			sendCounts = new int[k];
			sendDispls = new int[k + 1];
			
			makeGatherSendBuffer( sendBuffer, sendCounts, sendDispls, l);
			gather( sendBuffer, sendCounts, sendDispls, l);
		}
		
		// set inboundvalues
		if(rank==8){
		for(int s = 0; s<d; s++){
			for(int i=0; i<k; i++){

			System.out.println(String.format("l: %d, i: %d", s, i));
			for(int vid:gatherOrigin[k*s + i] ){
				System.out.print(String.format("%d ", vid));
			}
			}
		}}
		for( int vid: gatherBuffer.keySet() ){
			if(rank==8){System.out.println(String.format("vid: %d, val: %f", vid, gatherBuffer.get(vid)));}
			if(inboundMap.containsKey(vid)){
				inboundValues[inboundMap.get(vid)] = gatherBuffer.get(vid);
			}
		}
		
		
	}
	
	
	public void scatterConfig( int[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		int [] recvCounts = new int[k];
		int [] recvBuffer;

		for(int i = 0; i < k; i++){
			
			scatterDest[ k * level + i] = new LinkedList<Integer>();
			
			for(int j = 0; j<sendCounts[i]; j++){
				scatterDest[k * level + i].add(sendBuffer[sendDispls[i]+j]);
			}
			
			int right = getRight(i, level);
			int left = getLeft(i, level);
			
		//	if(rank==10)System.out.println(String.format("right: %d, left: %d, i: %d level: %d", right, left, i, level));	
			MPI.COMM_WORLD.Sendrecv(sendCounts, i, 1, MPI.INT, right, 0, recvCounts, i, 1, MPI.INT, left, 0);
				
			recvBuffer = new int[recvCounts[i]];
		//	if(rank==10)System.out.println(String.format("sc: %d, rc: %d, i: %d", sendCounts[i], recvCounts[i], i));	
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.INT, right, 0, recvBuffer, 0, recvCounts[i], MPI.INT, left, 0);

			
			scatterOrigin[ k * level + i] = new LinkedList<Integer>();
				
			for(int j = 0; j<recvCounts[i]; j++){
				scatterOrigin[k * level + i].add(recvBuffer[j]);
			}

			
		}
	}
	
	public void gatherConfig( int[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		int [] recvCounts = new int[k];
		int [] recvBuffer;		

		for(int i = 0; i < k; i++){
				
			int right = getRight(i, level);
			int left = getLeft(i, level);
				
			MPI.COMM_WORLD.Sendrecv(sendCounts, i, 1, MPI.INT, right, 0, recvCounts, i, 1, MPI.INT, left, 0);
					
			recvBuffer = new int[recvCounts[i]];
				
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.INT, right, 0, recvBuffer, 0, recvCounts[i], MPI.INT, left, 0);
			
			
			gatherDest[k * level + i] = new LinkedList<Integer>();
			gatherOrigin[k * level + (k-i)%k] = new LinkedList<Integer>();
				
			for(int j = 0; j<recvCounts[i]; j++){
				gatherDest[k * level + i].add(recvBuffer[j]);
			}
				
			for(int j = 0; j<sendCounts[i]; j++){
				gatherOrigin[k * level + (k-i)%k].add(sendBuffer[sendDispls[i] + j]);
			}
			
			
		}			
	}
	
	public void scatter( float[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		float [] buffer;
		
		for(int i = 0; i < k; i++){
			
			int right =  getRight(i, level);
			int left = getLeft(i, level);
			
			int bufferCount = scatterOrigin[k*level + i].size();
			buffer = new float[bufferCount];

			if( i !=0 ){
				
				MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.FLOAT, right, 0, buffer, 0, bufferCount, MPI.FLOAT, left, 0);
				int j = 0;
				for( int vid : scatterOrigin[k * level + i]){
					scatterBuffer.put( vid, scatterBuffer.get(vid) + buffer[j]);
					j++;
				}
				
			}else{
				
				if( level == -1){
					int j = 0;
					for( int vid : scatterOrigin[k * level + i]){
						scatterBuffer.put( vid, scatterBuffer.get(vid) + sendBuffer[sendDispls[i] + j]);
						j++;
					}
				}
			}
		};
	}
	
	public void gather( float[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		float [] recvBuffer;
	
		for(int i = 0; i < k; i++){
				
			int right =  getRight(i, level);
			int left = getLeft(i, level);
						
			if(i != 0){
				
				int bufferCount = gatherOrigin[k*level + i].size();
				recvBuffer = new float[bufferCount];
			
				MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.FLOAT, right, 0, recvBuffer, 0, bufferCount, MPI.FLOAT, left, 0);
			
				int j = 0;
				for( int vid : gatherOrigin[k*level + i]){
					gatherBuffer.put(vid, recvBuffer[j]);
					j++;
				}
			}else{
				
				int j = 0;
				for( int vid : gatherOrigin[k*level + i]){
					gatherBuffer.put(vid, sendBuffer[sendDispls[i] + j]);
					j++;
				}
			}
					
		}
	}
	
	public void terminate() throws MPIException{
		MPI.Finalize();
	}

}
