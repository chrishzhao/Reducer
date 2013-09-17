package Comm;

import java.lang.reflect.Array;
import java.util.*;

import mpi.*;

public class Reducer {
	
	// system setup parameters
	public int rank;
	public int size;
	
	public int[] hostIndices;
	
	// routing table for 1d, 2d case, maybe thinking to change it to hashmap
	private LinkedList<Integer> [] gatherDest;
	private LinkedList<Integer> [] scatterOrigin;
	
	// 2d setup
	private int hSize;
	private int vSize;
	private int hPos;
	private int vPos;
	private int scatterLength = 0;
	private int gatherLength = 0;
	// intermediate buffer store result after first stage, according to internalScatterBufferMap;
	private int[] internalScatterBuffer;
	// keep track of vertex index to buffer location mapping.
	private Map<Integer, Integer> internalScatterBufferMap;
	
	private int[] internalGatherBuffer;
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
		scatterOrigin = (LinkedList<Integer>[]) new LinkedList[size];
		
		setHostIndices();
	}
	
	public void init2d(){
		gatherDest = (LinkedList<Integer>[]) new LinkedList[size];
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
	
	
	public void config2d(int[] outboundIndices, int[]hostIndices, int[] inboundIndices, 
						int[] inboundCounts, int[] inboundDispls) throws MPIException {
		//this is wrong!! two directions are different!!
		scatterConfig2d(outboundIndices, hostIndices);

	}
	
	public void sparseAllReduce2d(float[] outboundValues, float[] hostValues, int[] inboundValues, 
			int[] inboundCounts, int[] inboundDispls) throws MPIException {
		scatter2d(outboundValues, hostValues);
	}
	
	public int getHostRank(int index){
		return index/size;
	}
	
	public void setHostIndices(){
		int hostSize = (modelSize+size)/size;
		for(int i = 0; i < hostSize; i++){
			hostIndices[i] = rank*hostSize + 1;
		}
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
		
		// add carry-over indices to the table
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
	
	// add host index to the set !!
	public void setScatterLength(){
		
		Set<Integer> vSet = new HashSet<Integer>();
		for(int i = 0; i < hSize; i++){
			
			int right =  getRightRank(i);
			vSet.addAll(scatterOrigin[right]);
			
		}
		
		// add host set
		for( int i : hostIndices){
			vSet.add(i);
		}
		
		scatterLength = vSet.size();
		
		internalScatterBuffer = new int[scatterLength];
		
	}
	
	// vertical scatter map (after horzontal scatter) add host index to the set !!
	public void setupScatterMapV(int[] sendBuffer, int[] sendCounts, int[] sendDispls){
		
		Set<Integer> vSet = new HashSet<Integer>();
		for(int i = 0; i < hSize; i++){
			
			int right =  getRightRank(i);
			vSet.addAll(scatterOrigin[right]);
			
		}
		for( int i : hostIndices){
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
		
		// allocate host nodes, should be no vertex assigned to other than vertical group according the pre-selection
		
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
		
		// 
		int sendBufferPointer = 0;
		for(int i = 0; i < hSize; i++){
			
			int down = getDownRank(i);
			int count = 0;
			
			sendDispls[i] = sendBufferPointer;
			
			for(int j = 0; j<vSize; j++){
				System.arraycopy(outboundIndices, sdisplsV[down], sendBuffer, sendBufferPointer, scountsV[down]);
				sendBufferPointer += scountsV[down];
				count += scountsV[down];
			}
			
			sendCounts[i] = count;
		}
		
		sendDispls[vSize] = outboundIndices.length;
		
		
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
	
	public void scatterConfig2d(int[] outboundIndices, int[] hostIndices) throws MPIException{
		
		int[] sendBufferH = new int[outboundIndices.length];
		int[] sendCountsH = new int[hSize];
		int[] sendDisplsH = new int[hSize+1];
		
		setupScatterMapH(outboundIndices, sendBufferH, sendCountsH, sendDisplsH);
		scatterConfig(sendBufferH, sendCountsH, sendDisplsH, 'h');
		
		
		setScatterLength();
		
		int[] sendBufferV = new int[scatterLength];
		int[] sendCountsV = new int[hSize];
		int[] sendDisplsV = new int[hSize+1];
		
		setupScatterMapV(sendBufferV, sendCountsV, sendDisplsV);
		scatterConfig(sendBufferV, sendCountsV, sendDisplsV, 'v');
	}
	
	/*
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
	
	*/
	
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
			
			int j = 0;
			for( float val : scatterOrigin[left]){
				internalScatterBuffer[internalScatterBufferMap.get(val)] += buffer[j];
				j++;
			}
		}
	}
	
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
				System.arraycopy(outboundValues, sdisplsV[right], sendBufferH, sendBufferPointerH, scountsV[right]);
				sendBufferPointerH += scountsV[right];
				count += scountsV[right];
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
			int count = 0;
			
			sendDisplsV[i] = sendBufferPointerV;
			
			for(int j = 0; j<vSize; j++){
				System.arraycopy(internalScatterBuffer, sdisplsV[down], sendBufferV, sendBufferPointerV, scountsV[down]);
				sendBufferPointerV += scountsV[down];
				count += scountsV[down];
			}
			
			sendCountsV[i] = count;
		}
		
		sendDisplsV[vSize] = outboundValues.length;
		
		scatter(sendBufferV, sendCountsV, sendDisplsV, 'v');
		
	}
	
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
		
		internalGatherBuffer = new int[gatherLength];
			
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
		
	public void gatherConfig2d(int[] inboundIndices, int[] hostIndices) throws MPIException{
		
		
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
	public void gather(float [] sendBuffer, float [] recvCounts, int [] recvDispls ) throws MPIException{
			
		int right = 0;
		int left = 0;
		float [] sendbuffer;
		float [] recvbuffer;
		int recvpointer = 0;
			
			//left = rank - 1;
			//if( left < 0 ){
			//	left += size;
			//}
		for(int i = 0; i<rank; i++){
			recvpointer += recvcounts[i];
		}
			
		for(int i = 0; i < size; i++){
				
			right = (rank + i) % size;
			left = rank - i;
			if( left < 0 ){
				left += size;
			}			
				
			int sendcount = gatherDest[left].size();
			sendbuffer = new float[sendcount];
				
			Iterator<Integer> itr = gatherDest[left].iterator();
			int j = 0;
			while(itr.hasNext()){
				int next = itr.next();
				int k = (next % dim_per_proc);
				sendbuffer[j] = sendbuf[k];
				j++;
			}
				
			recvbuffer = new float[recvcounts[right]];

			MPI.COMM_WORLD.Sendrecv(sendbuffer, 0, sendcount, MPI.FLOAT, left, 0, recvbuffer, 0, recvcounts[right], MPI.FLOAT, right, 0);
				
			for( j = 0; j < recvcounts[right]; j++){
				recvbuf[recvpointer % recvbuf.length] = recvbuffer[j];
				recvpointer++;
			}
		}
	}
		
		
	public void gather2d(float[] hostValues, float[] inboundValues){
		
		int[] recvCountsV = new int[hSize];
		int[] recvDisplsV = new int[hSize+1];
		
		int recvBufferPointerV = 0;
		for(int i = 0; i < vSize; i++){
			
			int down = getDownRank(i);
			int count = 0;
			
			sendDisplsH[i] = sendBufferPointerH;
			
			for(int j = 0; j<vSize; j++){
				System.arraycopy(outboundValues, sdisplsV[down], sendBufferH, sendBufferPointerH, scountsV[down]);
				sendBufferPointerH += scountsV[down];
				count += scountsV[down];
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
			int count = 0;
			
			sendDisplsV[i] = sendBufferPointerV;
			
			for(int j = 0; j<vSize; j++){
				System.arraycopy(internalScatterBuffer, sdisplsV[down], sendBufferV, sendBufferPointerV, scountsV[down]);
				sendBufferPointerV += scountsV[down];
				count += scountsV[down];
			}
			
			sendCountsV[i] = count;
		}
		
		sendDisplsV[vSize] = outboundValues.length;
		
		scatter(sendBufferV, sendCountsV, sendDisplsV, 'v');
		
	}
	

	

}
