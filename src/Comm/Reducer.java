package Comm;

import java.lang.reflect.Array;
import java.util.*;

import mpi.*;

public class Reducer {
	
	// system setup parameters
	public int rank;
	public int size;
	
	public int[] hostVertexIndices;
	
	// routing table for 1d, 2d case, maybe thinking to change it to hashmap
	private LinkedList<Integer> [] gatherDest;
	private LinkedList<Integer> [] gatherOrigin;
	private LinkedList<Integer> [] scatterOrigin;
	
	// 2d setup
	private int hSize;
	private int vSize;
	private int hPos;
	private int vPos;
	private int scatterLength = 0;
	private int gatherLength = 0;
	// intermediate buffer store result after first stage, according to internalScatterBufferMap;
	private float[] internalScatterBuffer;
	// keep track of vertex index to buffer location mapping.
	private Map<Integer, Integer> internalScatterBufferMap;
	
	private float[] internalGatherBuffer;
	private Map<Integer, Integer> internalGatherBufferMap;
	//private int[] hRange;
	//private int[] vRange;
	
	// keep track of scatter buffer arrangement
	private int[] scountsH;
	private int[] scountsV;
	private int[] sdisplsH;
	private int[] sdisplsV;
	
	private int[] rcountsH;
	private int[] rcountsV;
	private int[] rdisplsH;
	private int[] rdisplsV;

	// model parameter
	private int modelSize;

	// performance measure
	private int byteCount = 0;
	private long nanoTime = 0;
	private long sTime = 0;
	private long eTime = 0;
	
	public Reducer( String[] args ) throws MPIException{
		
		MPI.Init(args);
		rank = MPI.COMM_WORLD.Rank();
		size = MPI.COMM_WORLD.Size();
		modelSize = Integer.parseInt(args[0]);
	}
	
	public void init(){
		gatherDest = (LinkedList<Integer>[]) new LinkedList[size];
		gatherOrigin = (LinkedList<Integer>[]) new LinkedList[size];
		scatterOrigin = (LinkedList<Integer>[]) new LinkedList[size];
		
		
		setHostIndices();
	}
	
	public void init2d(){
		gatherDest = (LinkedList<Integer>[]) new LinkedList[size];
		gatherOrigin = (LinkedList<Integer>[]) new LinkedList[size];
		scatterOrigin = (LinkedList<Integer>[]) new LinkedList[size];
		
		setHostIndices();
		
		internalScatterBufferMap = new HashMap<Integer, Integer>();
		
		//currently only handle size is a square of an integer
		hSize = (int)Math.sqrt((double)size);
		vSize = (int)Math.sqrt((double)size);
		
		//hRange = new int[hSize];
		//vRange = new int[vSize];
		
		//row number
		vPos = rank / hSize;
		
		//column number
		hPos = rank % hSize;
		
		scountsH = new int[size];
		sdisplsH = new int[size+1];
		
		scountsV = new int[size];
		sdisplsV = new int[size+1];
		
		rcountsH = new int[size];
		rdisplsH = new int[size+1];
		
		rcountsV = new int[size];
		rdisplsV = new int[size+1];
		
		//for(int i = 0; i < hSize; i++){
		//	hRange[i] = h*hSize + i;
		//}
		
		//for(int i = 0; i < vSize; i++){
		//	vRange[i] = v + hSize*i;
		//}
	}
	
	// the wrapper function that calls scatter config and gather config
	public void config2d(int[] outboundIndices, int[] inboundIndices, int[]hostIndices) throws MPIException {

		scatterConfig2d(outboundIndices, hostIndices);
		//gatherConfig2d(inboundIndices);
		
		// need to return hostIndices

	}
	
	// the wrapper function that calls scatter and gather
	public void sparseAllReduce2d(float[] outboundValues, float[] inboundValues, float[] hostValues) throws MPIException {
		scatter2d(outboundValues, hostValues);
		//gather2d(hostValues, inboundValues);
		
		// need to return hostvalues
	}
	
	// return the rank of the machine that hosts the vertex
	public int getHostRank(int vertexIndex){
		int hostSize = (modelSize+size-1)/size;
		return vertexIndex/hostSize;
	}
	
	// return the set of vertex indices (in an array) the current machine host
	public void setHostIndices(){
		int hostSize = (modelSize+size-1)/size;
		for(int i = 0; i < hostSize; i++){
			hostVertexIndices[i] = rank*hostSize + i;
		}
	}
	
	// return the index of the vertex in current host buffer
	public int getHostIndex(int vertexIndex){
		return vertexIndex % ((modelSize+size-1)/size);
	}
	
	// return the rank of the right neighbour of current machine
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
	
	// return the rank of the right/down neighbour of a neighbour; 'h' for getting vertical neighbours for horizontal neighbours.
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
	
	// setup horizontal scatter map 
	// add carry-over indices to the map buffer
	// assuming outboundIndices are sorted according to host
	public void setupScatterMapH(int[] outboundIndices, int[] sendBuffer, int[] sendCounts, int[] sendDispls){
		
		int lRank = -1;
		int cRank = 0;

		for( int i = 0; i < outboundIndices.length; i++ ){
			cRank = getHostRank( outboundIndices[i] );
			if( lRank != cRank ){
				for( int j = lRank+1; j <= cRank; j++ ){
					sdisplsH[j] = i;
				}
			}
			lRank = cRank;
		}
		
		for( int i = lRank+1; i <= size; i++){
			sdisplsH[i] = outboundIndices.length;
		}
		
		for( int i = 0; i < size; i++){
			scountsH[i] = sdisplsH[i+1] - sdisplsH[i];
		}
		
		// add carry-over indices to the scatter map
		int sendBufferPointer = 0;
		for(int i = 0; i < hSize; i++){
			
			int right = getRightRank(i);
			int count = 0;
			
			sendDispls[i] = sendBufferPointer;
			
			for(int j = 0; j<vSize; j++){
				int nRank = getNeighbourRank(right, j, 'h');
				System.arraycopy(outboundIndices, sdisplsH[nRank], sendBuffer, sendBufferPointer, scountsH[nRank]);
				sendBufferPointer += scountsH[nRank];
				count += scountsH[nRank];
			}
			
			sendCounts[i] = count;
		}
		
		sendDispls[hSize] = outboundIndices.length;
		
	}
	
	// set the length of internalScatterBuffer
	public void setScatterLength(){
		
		Set<Integer> vSet = new HashSet<Integer>();
		for(int i = 0; i < hSize; i++){
			
			int right =  getRightRank(i);
			vSet.addAll(scatterOrigin[right]);
			
		}
		
		// add host set
		for( int i : hostVertexIndices){
			vSet.add(i);
		}
		
		scatterLength = vSet.size();
		
		internalScatterBuffer = new float[scatterLength];
		
	}
	
	// set up vertical scatter map, including carried-over vertices
	// construct vertex->index map for internalScatterBuffer
	public void setupScatterMapV(int[] sendBuffer, int[] sendCounts, int[] sendDispls){
		
		Set<Integer> vSet = new HashSet<Integer>();
		for(int i = 0; i < hSize; i++){
			
			int right =  getRightRank(i);
			vSet.addAll(scatterOrigin[right]);
			
		}
		for( int i : hostVertexIndices){
			vSet.add(i);
		}
		
		int[] outboundIndices = new int[scatterLength];
		
		int k = 0;
		for( Integer i : vSet){
			outboundIndices[k] = i;
			k++;
		}

		Arrays.sort(outboundIndices);
		
		// build vertex-index map
		
		for( int i = 0; i<outboundIndices.length; i++){
			internalScatterBufferMap.put(outboundIndices[i], i);
		}
		
		// tag boundaries of outboundIndices
		// no vertex should be assigned to other than vertical group due to pre-selection
		
		int lRank = -1;
		int cRank = 0;

		for( int i = 0; i < outboundIndices.length; i++ ){
			cRank = getHostRank( outboundIndices[i] );
			if( lRank != cRank ){
				for( int j = lRank+1; j <= cRank; j++ ){
					sdisplsV[j] = i;
				}
			}
			lRank = cRank;
		}
		
		for( int i = lRank+1; i <= size; i++){
			sdisplsV[i] = outboundIndices.length;
		}
		
		for( int i = 0; i < size; i++){
			scountsV[i] = sdisplsV[i+1] - sdisplsV[i];
		}
		
		// build sendBuffer
		int sendBufferPointer = 0;
		for(int i = 0; i < vSize; i++){
			
			int down = getDownRank(i);
			
			sendDispls[i] = sendBufferPointer;
			
			System.arraycopy(outboundIndices, sdisplsV[down], sendBuffer, sendBufferPointer, scountsV[down]);
			sendBufferPointer += scountsV[down];
			sendCounts[i] = scountsV[down];
			
		}
		
		sendDispls[vSize] = outboundIndices.length;
		
	}
	 
	// the basic element of scatter config
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
	
	// scatter config on a 2d mesh (butterfly network), first horizontal then vertical
	// for both horizontal and vertical, first setup scatter map, then scatter across machines
	// internalScatterBuffer is created for internal reduction
	public void scatterConfig2d(int[] outboundIndices, int[] hostIndices) throws MPIException{
		
		int[] sendBufferH = new int[outboundIndices.length];
		int[] sendCountsH = new int[hSize];
		int[] sendDisplsH = new int[hSize+1];
		
		setupScatterMapH(outboundIndices, sendBufferH, sendCountsH, sendDisplsH);
		scatterConfig(sendBufferH, sendCountsH, sendDisplsH, 'h');
		
		
		setScatterLength();
		
		int[] sendBufferV = new int[scatterLength];
		int[] sendCountsV = new int[vSize];
		int[] sendDisplsV = new int[vSize+1];
		
		setupScatterMapV(sendBufferV, sendCountsV, sendDisplsV);
		scatterConfig(sendBufferV, sendCountsV, sendDisplsV, 'v');
		
		hostIndices = hostVertexIndices.clone();
	}
	
	// basic element of scatter
	public void scatter(float [] sendBuffer, int [] sendCounts, int [] sendDispls, char orient) throws MPIException{

		float [] buffer;
		
		for(int i = 0; i < size; i++){
			
			int right =  (orient == 'h') ? getRightRank(i) : getDownRank(i);
			int left = (orient == 'h') ? getLeftRank(i) : getUpRank(i);
			
			int bufferCount = scatterOrigin[left].size();
			buffer = new float[bufferCount];
			
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[right], sendCounts[right], MPI.FLOAT, right, 0, buffer, 0, bufferCount, MPI.FLOAT, left, 0);
			
			/*
			Iterator<Integer> itr = scatterOrigin[left].iterator();
			int j = 0;
			while(itr.hasNext()){
				int next = itr.next();
				int k = (next % dim_per_proc);
				recvbuf[k] += buffer[j];
				j++;
			}*/
			
			// avoid duplicate addition from itself.
			if( i != 0 || orient != 'v'){
				int j = 0;
				for( float val : scatterOrigin[left]){
					internalScatterBuffer[internalScatterBufferMap.get(val)] += buffer[j];
					j++;
				}
			}
		}
	}
	
	// scatter on a 2d mesh (butterfly network), first horizontal then vertical
	public void scatter2d(float[] outboundValues, float[] hostValues)throws MPIException{
		
		float[] sendBufferH = new float[outboundValues.length];
		int[] sendCountsH = new int[hSize];
		int[] sendDisplsH = new int[hSize+1];
		
		int sendBufferPointerH = 0;
		for(int i = 0; i < hSize; i++){
			
			int right = getRightRank(i);
			int count = 0;
			
			sendDisplsH[i] = sendBufferPointerH;
			
			for(int j = 0; j<vSize; j++){
				int nRank = getNeighbourRank(right, j, 'h');
				System.arraycopy(outboundValues, sdisplsV[nRank], sendBufferH, sendBufferPointerH, scountsV[nRank]);
				sendBufferPointerH += scountsV[nRank];
				count += scountsV[nRank];
			}
			
			sendCountsH[i] = count;
		}
		
		scatter(sendBufferH, sendCountsH, sendDisplsH, 'h');

		
		float[] sendBufferV = new float[scatterLength];
		int[] sendCountsV = new int[hSize];
		int[] sendDisplsV = new int[hSize+1];
		
		int sendBufferPointerV = 0;
		for(int i = 0; i < hSize; i++){
			
			int down = getDownRank(i);

			sendDisplsV[i] = sendBufferPointerV;

			System.arraycopy(internalScatterBuffer, sdisplsV[down], sendBufferV, sendBufferPointerV, scountsV[down]);
			sendBufferPointerV += scountsV[down];
			
			sendCountsV[i] = scountsV[down];

		}
		
		sendDisplsV[vSize] = outboundValues.length;
		
		scatter(sendBufferV, sendCountsV, sendDisplsV, 'v');
		
		hostValues = new float[hostVertexIndices.length];
		
		for( int vi : hostVertexIndices){
			
			hostValues[getHostIndex(vi)] = internalScatterBuffer[internalScatterBufferMap.get(vi)];
		}
		
	}
	/*
	// horizontal gather map (assuming outboundIndices are sorted according to host rank) 
	public void setupGatherMapV(int[] inboundIndices, int[] sendBuffer, int[] sendCounts, int[] sendDispls){
			
		int lRank = -1;
		int cRank = 0;

		for( int i = 0; i < inboundIndices.length; i++ ){
			cRank = getHostRank( inboundIndices[i] );
			if( lRank != cRank ){
				for( int j = lRank+1; j <= cRank; j++ ){
					rdisplsV[j] = i;
				}
			}
			lRank = cRank;
		}
			
		for( int i = lRank+1; i <= size; i++){
			rdisplsV[i] = inboundIndices.length;
		}
			
		for( int i = 0; i < size; i++){
			rcountsV[i] = rdisplsV[i+1] - rdisplsV[i];
		}
			
		// add carry-over indices to the table
		int sendBufferPointer = 0;
		for(int i = 0; i < vSize; i++){
				
			int down = getDownRank(i);
			int count = 0;
				
			sendDispls[i] = sendBufferPointer;
				
			for(int j = 0; j<hSize; j++){
				int nRank = getNeighbourRank(down, j, 'v');
				System.arraycopy(inboundIndices, rdisplsV[nRank], sendBuffer, sendBufferPointer, rcountsV[nRank]);
				sendBufferPointer += rcountsV[nRank];
				count += rcountsV[nRank];
			}
				
			sendCounts[i] = count;
		}
			
		sendDispls[hSize] = inboundIndices.length;
			
	}
		
	// 
	public void setGatherLength(){
			
		Set<Integer> vSet = new HashSet<Integer>();
		for(int i = 0; i < vSize; i++){
				
			int down =  getDownRank(i);
			vSet.addAll(gatherDest[down]);
				
		}
			
			
		gatherLength = vSet.size();
		
		internalGatherBuffer = new float[gatherLength];
			
	}
		
	// vertical gather map (after horzontal scatter) add host index to the set !!
	public void setupGatherMapH(int[] sendBuffer, int[] sendCounts, int[] sendDispls){
			
		Set<Integer> vSet = new HashSet<Integer>();
		for(int i = 0; i < hSize; i++){
				
			int right =  getRightRank(i);
			vSet.addAll(gatherDest[right]);				
		}
			
		int[] inboundIndices = new int[gatherLength];
		
		int k = 0;
		for( Integer i : vSet){
			inboundIndices[k] = i;
			k++;
		}

		Arrays.sort(inboundIndices);
		
		// build vertex-index map
		
		for( int i = 0; i<inboundIndices.length; i++){
			internalGatherBufferMap.put(inboundIndices[i], i);
		}
		
			
		// allocate host nodes, should be no vertex assigned to other than vertical group according the pre-selection
			
		int lRank = -1;
		int cRank = 0;

		for( int i = 0; i < inboundIndices.length; i++ ){
			cRank = getHostRank( inboundIndices[i] );
			if( lRank != cRank ){
				for( int j = lRank+1; j <= cRank; j++ ){
					rdisplsH[j] = i;
				}
			}
			lRank = cRank;
		}
			
		for( int i = lRank+1; i <= size; i++){
			rdisplsH[i] = inboundIndices.length;
		}
			
		for( int i = 0; i < size; i++){
			rcountsH[i] = rdisplsH[i+1] - rdisplsH[i];
		}
			
			// 
		int sendBufferPointer = 0;
		for(int i = 0; i < vSize; i++){
				
			int right = getRightRank(i);
			int count = 0;
				
			sendDispls[i] = sendBufferPointer;
				
			for(int j = 0; j<hSize; j++){
				System.arraycopy(inboundIndices, rdisplsH[right], sendBuffer, sendBufferPointer, rcountsH[right]);
				sendBufferPointer += rcountsH[right];
				count += rcountsH[right];
			}
				
			sendCounts[i] = count;
		}
			
		sendDispls[vSize] = inboundIndices.length;
				
	}	
	
	// the basic gather config element
	public void gatherConfig(int[] sendBuffer, int[] sendCounts, int[] sendDispls, char orient) throws MPIException{
			
		int [] recvCounts = new int[size];
		int [] recvBuffer;
		
		for(int i = 0; i < hSize; i++){
				
			int right =  (orient == 'h') ? getRightRank(i) : getDownRank(i);
			int left = (orient == 'h') ? getLeftRank(i) : getUpRank(i);
				
			MPI.COMM_WORLD.Sendrecv(sendCounts, i, 1, MPI.INT, right, 0, recvCounts, left, 1, MPI.INT, left, 0);
					
			recvBuffer = new int[recvCounts[left]];
				
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[i], sendCounts[i], MPI.INT, right, 0, recvBuffer, 0, recvCounts[left], MPI.INT, left, 0);
							
			gatherDest[left] = new LinkedList<Integer>();
			for(int j = 0; j<recvCounts[left]; j++){
				gatherDest[left].add(recvBuffer[j]);
			}
		}			
	}
		
	public void gatherConfig2d(int[] inboundIndices) throws MPIException{
		
		
		int[] sendBufferV = new int[inboundIndices.length];
		int[] sendCountsV = new int[vSize];
		int[] sendDisplsV = new int[vSize+1];
			
		setupGatherMapV(inboundIndices, sendBufferV, sendCountsV, sendDisplsV);
		gatherConfig(sendBufferV, sendCountsV, sendDisplsV, 'v');
			
		setGatherLength();
		
		int[] sendBufferH = new int[gatherLength];
		int[] sendCountsH = new int[hSize];
		int[] sendDisplsH = new int[hSize+1];
			
		setupGatherMapH(sendBufferH, sendCountsH, sendDisplsH);
		gatherConfig(sendBufferH, sendCountsH, sendDisplsH, 'h');
				
	}
	
	// for both gather and scatter try not to sendrecv to myself; handle i = 0 separately.
	public void gather(float [] sendBuffer, int [] sendCounts, int [] sendDispls, char orient ) throws MPIException{
			
		float [] recvBuffer;
		
		for(int i = 0; i < vSize; i++){
				
			int right =  (orient == 'h') ? getRightRank(i) : getDownRank(i);
			int left = (orient == 'h') ? getLeftRank(i) : getUpRank(i);
			
			int bufferCount = 0;
			
			for(int j = 0; j<hSize; j++){
				int nRank = getNeighbourRank(right, j, 'v');
				bufferCount += rcountsV[nRank];
			}
			
			recvBuffer = new float[bufferCount];
			
			MPI.COMM_WORLD.Sendrecv(sendBuffer, sendDispls[right], sendCounts[right], MPI.FLOAT, right, 0, recvBuffer, 0, bufferCount, MPI.FLOAT, left, 0);
			
			int j = 0;
			for( float val : gatherOrigin[left]){
				internalGatherBuffer[internalGatherBufferMap.get(val)] = recvBuffer[j];
				j++;
			}
					
		}
	}
			
	public void gather2d(float[] hostValues, float[] inboundValues) throws MPIException {
		
		
		float[] sendBufferV;
		int[] sendCountsV = new int[vSize];
		int[] sendDisplsV = new int[vSize + 1];
		
		int sendBufferVL = 0;
		
		for(int i = 0; i<vSize; i++){
			int down = getDownRank(i);
			sendBufferVL += gatherDest[down].size();
		}
		
		sendBufferV = new float[sendBufferVL];
		
		int bufferPointerV = 0;
		
		for(int i = 0; i<vSize; i++){
			int down = getDownRank(i);
			sendDisplsV[i] = bufferPointerV;
			for( int vid : gatherDest[down]){
				sendBufferV[bufferPointerV] = hostValues[getHostIndex(vid)];
				bufferPointerV += 1;
				sendCountsV[i] += 1;
			}
		}
		
		gather(sendBufferV, sendCountsV, sendDisplsV, 'v');

		float[] sendBufferH;
		int[] sendCountsH = new int[hSize];
		int[] sendDisplsH = new int[hSize + 1];
		
		int sendBufferHL = 0;
		
		for(int i = 0; i<hSize; i++){
			int right = getRightRank(i);
			sendBufferHL += gatherDest[right].size();
		}
		
		sendBufferH = new float[sendBufferHL];
		
		int bufferPointerH = 0;
		
		for(int i = 0; i<vSize; i++){
			int right = getRightRank(i);
			sendDisplsH[i] = bufferPointerH;
			for( int vid : gatherDest[right]){
				sendBufferH[bufferPointerH] = internalGatherBuffer[internalGatherBufferMap.get(vid)];
				bufferPointerH += 1;
				sendCountsH[i] += 1;
			}
		}
		gather(sendBufferH, sendCountsH, sendDisplsH, 'h');
		
	}
	
*/

}
