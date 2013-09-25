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
	
	private LinkedList<IVec> scheduleVertexSets;
	private LinkedList<IVec> maps;
	
	//
	private IVec scatterVertexSet;
	private IVec gatherVertexSet;
	
	private float[] internalBuffer;
	
	// internal buffers
	
	// benchmarks
	private long commTime = 0;
	private long bufferTime = 0;
	private long bytes = 0;
	
	private long realCommTime = 0;
	private long bufferCommTime = 0;
	
	private long packingTime = 0;
	private long barrierTime = 0;
	
	private long configMergeTime = 0;
	private long configPartTime = 0;
	private long configCommTime = 0;
	
	public Reducer( String[] args ) throws MPIException{
		
		MPI.Init(args);
		rank = MPI.COMM_WORLD.Rank();
		size = MPI.COMM_WORLD.Size();
		modelSize = Integer.parseInt(args[0]);
		k = Integer.parseInt(args[1]);
		d = Integer.parseInt(args[2]);
	}
	
		
	public void init(){
		scheduleVertexSets = new LinkedList<IVec>();
		maps = new LinkedList<IVec>();
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
	
	public int getScatterOriginIndex(int i, int level){
		// offset level
		int offset = 2 * k * level;
		// offset dest
		offset += k;
		return offset + i;
	}
	
	public int getScatterDestIndex(int i, int level){
		int offset = 2 * k * level;
		return offset + i;
	}
	
	public int getGatherOriginIndex(int i, int level){
		// offset scatter
		int offset = 2 * k * d;
		// offset level
		offset += 2 * k * level;
		return offset + (k-i)%k;
	}
	
	public int getGatherDestIndex(int i, int level){
		// offset scatter
		int offset = 2 * k * d;
		// offset level
		offset += 2 * k * level;
		// offset origin
		offset += k;
		return offset + (k-i)%k;
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
	
	// need to change
	/*
	public int getGatherDest( int host, int level){
		
		for( int dest = 0; dest < k; dest ++){
			int right = getRight(dest, level);
			if((right / (int)Math.pow(k, level)) == (host / (int)Math.pow(k, level))){
				return dest;
			}
		}
		return -1;
	}*/
	public int getGatherDest( int host, int level){
		
		for( int dest = 0; dest < k; dest ++){
			int right = getRight(dest, level);
			if((right % (int)Math.pow(k, level+1)) == (host % (int)Math.pow(k, level+1))){
				return dest;
			}
		}
		return -1;
	}
	
	public void makeScatterSendBuffer( float[] sendBuffer, int[] sendCounts, int[] sendDispls, int level){
		
		int bufferPointer = 0;
		long sTime = System.nanoTime();
		
		for (int i = 0; i<k; i++){
			
			int index = getScatterDestIndex(i,level);
			int[] map = maps.get(index+1).data;
			
			sendDispls[i] = bufferPointer;
			for(int j=0; j<map.length; j++){
				sendBuffer[bufferPointer++] = internalBuffer[map[j]];
			}
			sendCounts[i] = map.length;
			
		}
		
		long eTime = System.nanoTime();
		packingTime += eTime - sTime;
		
	}
	
	public void makeGatherSendBuffer( float[] sendBuffer, int[] sendCounts, int[] sendDispls, int level){
		
		int bufferPointer = 0;
		long sTime = System.nanoTime();
		
		for (int i = 0; i<k; i++){
			
			int index = getGatherDestIndex(i,level);
			int[] map = maps.get(index+1).data;
			
			sendDispls[i] = bufferPointer;
			for(int j=0; j<map.length; j++){
				sendBuffer[bufferPointer++] = internalBuffer[map[j]];
			}
			sendCounts[i] = map.length;
			
		}
		
		long eTime = System.nanoTime();
		packingTime += eTime - sTime;
		
		
	}
	
	public void makeScatterConfigSendBuffer( int[] sendBuffer, int[] sendCounts, int[] sendDispls, int level){
		
		// sorting according to host rank
		// Arrays.sort(outboundIndices);
		int[] outboundIndices = scatterVertexSet.data;

		long sTime = System.nanoTime();
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
		long eTime = System.nanoTime();
		configPartTime += eTime - sTime;
		// add to scatter dest
		for( int i = 0; i<k; i++){
			int[] data = new int[sendCounts[i]]; 
			for( int j = 0; j<sendCounts[i]; j++){
				data[j] = sendBuffer[sendDispls[i] + j];
			}
			scheduleVertexSets.add(new IVec(data));
		}
		
		
		
	}
	
	public void makeGatherConfigSendBuffer( int[] sendBuffer, int[] sendCounts, int[] sendDispls, int level){
		
		int[] inboundIndices = gatherVertexSet.data;
		
		// sorting according to host rank
		//Arrays.sort(inboundIndices);
		long sTime = System.nanoTime();
		
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
		
		long eTime = System.nanoTime();
		configPartTime += eTime - sTime;
		// add to gather origin
		for( int i = 0; i<k; i++){
			int[] data = new int[sendCounts[i]]; 
			for( int j = 0; j<sendCounts[i]; j++){
				data[j] = sendBuffer[sendDispls[i] + j];
			}
			scheduleVertexSets.add(new IVec(data));
		}
	}
	
	public void config( int[] outboundIndices, int[] inboundIndices) throws MPIException{
		
		scatterVertexSet = new IVec(outboundIndices);
		gatherVertexSet = new IVec(inboundIndices);
		
		int [] sendBuffer;
		int [] sendCounts;
		int [] sendDispls;
		
		//System.out.println("scatter config");
		for( int l = 0; l < d; l++){
			
		//	if(rank == 10)System.out.println(String.format("rank: %d, l: %d", rank, l));
			sendBuffer = new int[scatterVertexSet.size()];
			sendCounts = new int[k];
			sendDispls = new int[k + 1];
			
			makeScatterConfigSendBuffer( sendBuffer, sendCounts, sendDispls, l);
		
		//	if(rank == 10)System.out.println(String.format("cp 1 rank: %d, l: %d", rank, l));
			scatterConfig( sendBuffer, sendCounts, sendDispls, l);
			
		//	if(rank == 10)System.out.println(String.format("cp 2 rank: %d, l: %d", rank, l));

			
		}
		
		//System.out.println("gather config");
		for( int l = 0; l<d; l++){
			//if(rank == 10)System.out.println(String.format("rank: %d, l: %d", rank, l));
			sendBuffer = new int[gatherVertexSet.size()];
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
			

		}
		
		long sTime = System.nanoTime();
		scheduleVertexSets.add(new IVec(outboundIndices));
		scheduleVertexSets.add(new IVec(inboundIndices));
		maps = IVec.mergeAndMap(scheduleVertexSets);
		internalBuffer = new float[maps.get(0).size()];
		long eTime = System.nanoTime();
		configMergeTime += eTime - sTime;
			
	}
	
	public void reduce( float[] outboundValues, float[] inboundValues) throws MPIException{
		
		// benchmark
		commTime = 0;
		bufferTime = 0;
		bytes = 0;
		realCommTime = 0;
		bufferCommTime = 0;
		packingTime = 0;
		barrierTime = 0;
		
		// get map for outbound vertices
		int[] omap = maps.get(1 + 4 * k * d).data;
		for(int j = 0; j<omap.length; j++){
			internalBuffer[omap[j]] = outboundValues[j];
		}
		
		// scatter
		float [] sendBuffer;
		int [] sendCounts;
		int [] sendDispls;
		
		for( int l = 0; l<d; l++){
			
			int bufferSize = 0;
			for(int i = 0; i < k; i++){
				int index = getScatterDestIndex(i,l);
				int[] vertices = scheduleVertexSets.get(index).data;
				bufferSize += vertices.length;
			}
			
			//if(rank == 0){System.out.println(String.format("scatter level %d: bufferSize: %d", l, bufferSize));}
			
			sendBuffer = new float[bufferSize];
			sendCounts = new int[k];
			sendDispls = new int[k + 1];
			
			long sTime = System.nanoTime();
			makeScatterSendBuffer( sendBuffer, sendCounts, sendDispls, l);
			long bTime = System.nanoTime();
			//MPI.COMM_WORLD.Barrier();
			long eTime = System.nanoTime();
			barrierTime += eTime - bTime;
			bufferTime += bTime - sTime;
			scatter( sendBuffer, sendCounts, sendDispls, l);
			sTime = System.nanoTime();
			commTime += sTime - eTime;
		}
		
		
		// gather
		for( int l = d-1; l>=0; l--){
			
			int bufferSize = 0;
			for(int i = 0; i < k; i++){
				int index = getGatherDestIndex(i,l);
				int[] vertices = scheduleVertexSets.get(index).data;
				bufferSize += vertices.length;
			}
			
			//if(rank == 0){System.out.println(String.format("gather level %d: bufferSize: %d", l, bufferSize));}
			
			sendBuffer = new float[bufferSize];
			sendCounts = new int[k];
			sendDispls = new int[k + 1];
			
			long sTime = System.nanoTime();
			makeGatherSendBuffer( sendBuffer, sendCounts, sendDispls, l);
			long bTime = System.nanoTime();
			//MPI.COMM_WORLD.Barrier();
			long eTime = System.nanoTime();
			barrierTime += eTime - bTime;
			bufferTime += bTime - sTime;
			gather( sendBuffer, sendCounts, sendDispls, l);
			sTime = System.nanoTime();
			commTime += sTime - eTime;
		}
		
		// get map for inbound vertices
		int[] imap = maps.get(1 + 4 * k * d + 1).data;
		for(int j = 0; j<imap.length; j++){
			inboundValues[j] = internalBuffer[imap[j]];
		}
		
	}
	
	
	public void scatterConfig( int[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		int [] recvCounts = new int[k];
		int [] recvBuffer;

		for(int i = 0; i < k; i++){
			
			int right = getRight(i, level);
			int left = getLeft(i, level);
			
			long sTime = System.nanoTime();
		//	if(rank==10)System.out.println(String.format("right: %d, left: %d, i: %d level: %d", right, left, i, level));	
			MPI.COMM_WORLD.Sendrecv(sendCounts, i, 1, MPI.INT, right, 0, recvCounts, i, 1, MPI.INT, left, 0);
				
			recvBuffer = new int[recvCounts[i]];
		    //System.out.println(String.format("sc: %d, rc: %d, i: %d", sendCounts[i], recvCounts[i], i));	
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.INT, right, 0, recvBuffer, 0, recvCounts[i], MPI.INT, left, 0);

			long eTime = System.nanoTime();
			configCommTime += eTime - sTime;
			scheduleVertexSets.add(new IVec(recvBuffer)); 
			scatterVertexSet = IVec.merge(scatterVertexSet, new IVec(recvBuffer));
			sTime = System.nanoTime();
			configMergeTime += sTime - eTime;
		}
	}
	
	public void gatherConfig( int[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		int [] recvCounts = new int[k];
		int [] recvBuffer;		

		for(int i = 0; i < k; i++){
				
			int right = getRight(i, level);
			int left = getLeft(i, level);
				
			long sTime = System.nanoTime();
			
			MPI.COMM_WORLD.Sendrecv(sendCounts, i, 1, MPI.INT, right, 0, recvCounts, i, 1, MPI.INT, left, 0);
					
			recvBuffer = new int[recvCounts[i]];
			//System.out.println(String.format("sc: %d, rc: %d, i: %d", sendCounts[i], recvCounts[i], i));	
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.INT, right, 0, recvBuffer, 0, recvCounts[i], MPI.INT, left, 0);

			long eTime = System.nanoTime();
			configCommTime += eTime - sTime;
			
			scheduleVertexSets.add(new IVec(recvBuffer));
			gatherVertexSet = IVec.merge(gatherVertexSet, new IVec(recvBuffer));
			sTime = System.nanoTime();
			configMergeTime += sTime - eTime;
				
			
		}			
	}
	
	public void scatter( float[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		float [] buffer;
		
		for(int i = 0; i < k; i++){
			
			int right =  getRight(i, level);
			int left = getLeft(i, level);
			
			int index = getScatterOriginIndex(i, level);
			int[] map = maps.get(index+1).data;
			
			if( i !=0 ){
				
				int bufferCount = map.length;
				buffer = new float[bufferCount];
				//System.out.println(String.format("sc: %d, rc: %d, i: %d", sendCounts[i], bufferCount, i));	
				long sTime = System.nanoTime();
				MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.FLOAT, right, 0, buffer, 0, bufferCount, MPI.FLOAT, left, 0);
				long eTime = System.nanoTime();
				realCommTime += eTime - sTime;
				//System.out.println(eTime-sTime);
	
				for(int j = 0; j<bufferCount; j++){
					internalBuffer[map[j]] += buffer[j];
				}
				sTime = System.nanoTime();
				bufferCommTime += sTime - eTime;
				bytes += 4*(sendCounts[i] + bufferCount);
				
			}else{
				
			}
		};
	}
	
	public void gather( float[] sendBuffer, int[] sendCounts, int[] sendDispls, int level) throws MPIException{
		
		float [] recvBuffer;
	
		for(int i = 0; i < k; i++){
				
			int right =  getRight(i, level);
			int left = getLeft(i, level);
						
			int index = getGatherOriginIndex(i, level);
			int[] map = maps.get(index+1).data;
			
			if(i != 0){
				int bufferCount = map.length;
				recvBuffer = new float[bufferCount];
			
				//System.out.println(String.format("sc: %d, rc: %d, i: %d", sendCounts[i], bufferCount, i));
				long sTime = System.nanoTime();
				MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.FLOAT, right, 0, recvBuffer, 0, bufferCount, MPI.FLOAT, left, 0);
				long eTime = System.nanoTime();
				realCommTime += eTime - sTime;
				//System.out.println(eTime-sTime);
				
				for( int j = 0; j<bufferCount; j++){
					internalBuffer[map[j]] = recvBuffer[j];
				}
				
				sTime = System.nanoTime();
				bufferCommTime += sTime - eTime;
				bytes += 4*(sendCounts[i] + bufferCount);
			}else{
				
				long sTime = System.nanoTime();
				for( int j = 0; j<sendCounts[i]; j++){
					internalBuffer[map[j]] = sendBuffer[sendDispls[i]+j];
				}
				long eTime = System.nanoTime();
				bufferCommTime += eTime - sTime;
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
	
	public float getPackingTime(){
		return packingTime/1000000000f;
	}
	
	public float getBarrierTime(){
		return barrierTime/1000000000f;
	}
	
	public float getConfigCommTime(){
		return configCommTime/1000000000f;
	}
	
	public float getConfigMergeTime(){
		return configMergeTime/1000000000f;
	}
	
	public float getConfigPartTime(){
		return configPartTime/1000000000f;
	}
	public void terminate() throws MPIException{
		MPI.Finalize();
	}

}


