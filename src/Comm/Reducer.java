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
	
	// scatterDest[l*k + i] stores the vertex id to send at the ith round scatter communication at level l, 
	//the ith round communication send to machine getRight(i, l);
	private LinkedList<Integer> [] scatterDest;
	// scatterOrigin[l*k + i] stores the vertex id to receive at the ith round scatter communication at level l, 
	//the ith round communication receive from machine getLeft(i, l);
	private LinkedList<Integer> [] scatterOrigin;
	// gatherDest[l*k + i] stores the vertex id to send at the ith round gather communication at level l, 
	//the ith round communication send to machine getRight(i, l);
	private LinkedList<Integer> [] gatherDest;
	// gatherOrigin[l*k + i] stores the vertex id to receive at the ith round gather communication at level l, 
	//the ith round communication receive from machine getLeft(i, l);
	private LinkedList<Integer> [] gatherOrigin;	
	
	// internal buffers
	// a union of scatter origins (including from itself i.e. outbound vertices)
	private Map<Integer, Float> scatterBuffer;
	// a union of gather origins (including to itself i.e. host vertices)
	private Map<Integer, Float> gatherBuffer;
	private Map<Integer, Float> hostBuffer;
	
	// vid -> buffer index map
	private Map<Integer, Integer> outboundMap;
	private Map<Integer, Integer> inboundMap;
	
	// benchmarks
	private long sTime;
	private long eTime;
	
	private long commTime = 0;
	private long bufferTime = 0;
	private long bytes = 0;
	
	private long realCommTime = 0;
	private long bufferCommTime = 0;
	
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
		
		scatterDest = (LinkedList<Integer>[]) new LinkedList[d * size];
		scatterOrigin = (LinkedList<Integer>[]) new LinkedList[d * size];
		gatherDest = (LinkedList<Integer>[]) new LinkedList[d * size];
		gatherOrigin = (LinkedList<Integer>[]) new LinkedList[d * size];
		
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
	
	// scatter destination at level "level" for vertex hosted by "host"
	// return -1 if don't need to scatter
	public int getScatterDest( int host, int level){
		
		for( int dest = 0; dest < k; dest ++){
			int right = getRight(dest, level);
			if((right % (int)Math.pow(k, level+1)) == (host % (int)Math.pow(k, level+1))){
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
			sendCounts[i] = scatterDest[level * size + getRight(i, level)].size();
			for(int vid: scatterDest[level * size + getRight(i, level)]){
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
			sendCounts[i] = gatherDest[level * size + getRight(i, level)].size();
			for(int vid: gatherDest[level * size + getRight(i, level)]){
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
				for( int vid: scatterOrigin[ l * size + getLeft(i, l)]){
					scatterBuffer.put( vid, 0f );
				}
				for( int vid: scatterDest[ l * size + getRight(i, l)]){
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
				for( int vid: gatherOrigin[ l * size + getLeft(i, l)]){
					gatherBuffer.put( vid, 0f );
				}
				for( int vid: gatherDest[ l * size + getRight(i, l)]){
					gatherBuffer.put( vid, 0f );
				}
			}
		}
		
		
	}
	
	public void reduce( float[] outboundValues, float[] inboundValues) throws MPIException{
		
		commTime = 0;
		bufferTime = 0;
		bytes = 0;
		realCommTime = 0;
		bufferCommTime = 0;
		
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
				bufferSize += scatterDest[l * size + getRight(i, l)].size();
			}
			sendBuffer = new float[bufferSize];
			sendCounts = new int[k];
			sendDispls = new int[k + 1];
			
			//sTime = System.nanoTime();
			makeScatterSendBuffer( sendBuffer, sendCounts, sendDispls, l);
			//eTime = System.nanoTime();
			//bufferTime += eTime - sTime;
			scatter( sendBuffer, sendCounts, sendDispls, l);
			//sTime = System.nanoTime();
			//commTime += sTime - eTime;
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
				bufferSize += gatherDest[l * size + getRight(i, l)].size();
			}
			
			sendBuffer = new float[bufferSize];
			sendCounts = new int[k];
			sendDispls = new int[k + 1];
			
			//sTime = System.nanoTime();
			makeGatherSendBuffer( sendBuffer, sendCounts, sendDispls, l);
			//eTime = System.nanoTime();
			//bufferTime += eTime - sTime;
			gather( sendBuffer, sendCounts, sendDispls, l);
			//sTime = System.nanoTime();
			//commTime += sTime - eTime;
		}
		
		// set inboundvalues
		//if(rank==8){
		//for(int s = 0; s<d; s++){
		//	for(int i=0; i<k; i++){

		//	System.out.println(String.format("l: %d, i: %d", s, i));
		//	for(int vid:gatherOrigin[k*s + i] ){
		//		System.out.print(String.format("%d ", vid));
		//	}
		//	}
		//}}
		for( int vid: gatherBuffer.keySet() ){
			//if(rank==8){System.out.println(String.format("vid: %d, val: %f", vid, gatherBuffer.get(vid)));}
			if(inboundMap.containsKey(vid)){
				inboundValues[inboundMap.get(vid)] = gatherBuffer.get(vid);
			}
		}
		
		
	}
	
	
	public void scatterConfig( int[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		int [] recvCounts = new int[k];
		int [] recvBuffer;

		for(int i = 0; i < k; i++){
			
			
			int right = getRight(i, level);
			int left = getLeft(i, level);
			
			scatterDest[ size * level + right] = new LinkedList<Integer>();
			
			for(int j = 0; j<sendCounts[i]; j++){
				scatterDest[size * level + right].add(sendBuffer[sendDispls[i]+j]);
			}

			
		//	if(rank==10)System.out.println(String.format("right: %d, left: %d, i: %d level: %d", right, left, i, level));	
			MPI.COMM_WORLD.Sendrecv(sendCounts, i, 1, MPI.INT, right, 0, recvCounts, i, 1, MPI.INT, left, 0);
				
			recvBuffer = new int[recvCounts[i]];
		    //System.out.println(String.format("sc: %d, rc: %d, i: %d", sendCounts[i], recvCounts[i], i));	
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.INT, right, 0, recvBuffer, 0, recvCounts[i], MPI.INT, left, 0);

			
			scatterOrigin[ size * level + left] = new LinkedList<Integer>();
				
			for(int j = 0; j<recvCounts[i]; j++){
				scatterOrigin[size * level + left].add(recvBuffer[j]);
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
			//System.out.println(String.format("sc: %d, rc: %d, i: %d", sendCounts[i], recvCounts[i], i));	
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.INT, right, 0, recvBuffer, 0, recvCounts[i], MPI.INT, left, 0);
			
			
			gatherDest[size * level + left] = new LinkedList<Integer>();
			gatherOrigin[size * level + right] = new LinkedList<Integer>();
				
			for(int j = 0; j<recvCounts[i]; j++){
				gatherDest[size * level + left].add(recvBuffer[j]);
			}
				
			for(int j = 0; j<sendCounts[i]; j++){
				gatherOrigin[size * level + right].add(sendBuffer[sendDispls[i] + j]);
			}
			
			
		}			
	}
	
	public void scatter( float[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		float [] buffer;
		
		for(int i = 0; i < k; i++){
			
			int right =  getRight(i, level);
			int left = getLeft(i, level);
			
			int bufferCount = scatterOrigin[size*level + left].size();
			buffer = new float[bufferCount];

			if( i !=0 ){
				//System.out.println(String.format("sc: %d, rc: %d, i: %d", sendCounts[i], bufferCount, i));	
				sTime = System.nanoTime();
				MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.FLOAT, right, 0, buffer, 0, bufferCount, MPI.FLOAT, left, 0);
				eTime = System.nanoTime();
				realCommTime += eTime - sTime;
				int j = 0;
				for( int vid : scatterOrigin[size * level + left]){
					scatterBuffer.put( vid, scatterBuffer.get(vid) + buffer[j]);
					j++;
				}
				sTime = System.nanoTime();
				bufferCommTime += sTime - eTime;
				bytes += 4*(sendCounts[i] + bufferCount);
				
			}else{
				
				if( level == -1){
					int j = 0;
					for( int vid : scatterOrigin[size * level + left]){
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
				
				int bufferCount = gatherOrigin[size*level + left].size();
				recvBuffer = new float[bufferCount];
			
				//System.out.println(String.format("sc: %d, rc: %d, i: %d", sendCounts[i], bufferCount, i));
				sTime = System.nanoTime();
				MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.FLOAT, right, 0, recvBuffer, 0, bufferCount, MPI.FLOAT, left, 0);
				eTime = System.nanoTime();
				realCommTime += eTime - sTime;
				int j = 0;
				for( int vid : gatherOrigin[size*level + left]){
					gatherBuffer.put(vid, recvBuffer[j]);
					j++;
				}
				sTime = System.nanoTime();
				bufferCommTime += sTime - eTime;
				bytes += 4*(sendCounts[i] + bufferCount);
			}else{
				
				int j = 0;
				for( int vid : gatherOrigin[size*level + left]){
					gatherBuffer.put(vid, sendBuffer[sendDispls[i] + j]);
					j++;
				}
			}
					
		}
	}
	
	public float getCommTime(){
		return commTime/1000000000f;
	}
	
	public float getBufferTime(){
		return bufferTime/1000000000f;
	}
	public float getThroughput(){
		return (float)bytes / realCommTime;
	}
	
	public float getRealCommTime(){
		return realCommTime/1000000000f;
	}
	
	public float getBufferCommTime(){
		return bufferCommTime/1000000000f;
	}
	public void terminate() throws MPIException{
		MPI.Finalize();
	}

}
