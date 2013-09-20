import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._

import mpi._
import sparsecomm._

class Model(args: Array[String]){

	var inboundIndices: Array[Int] = null;
	var inboundValues: Array[Float] = null;
	var outboundIndices: Array[Int] = null;
	var outboundValues: Array[Float] = null;
	var in: FMat = null;
	var out: FMat = null;
	
	var inMap: Map[Int, Int] = null;
	var outMap: Map[Int, Int] = null;
	var edgeMat: SMat = null;
	
	var nr: Int = -1;
	var nc: Int = -1;
	
	var vector: Array[Float] = null;
	
	var comm: SparseComm = null;
	
	var nvertices: Int = -1;
	var size: Int = -1;
	var nv_per_proc: Int = -1;

	
	var inbound_displs: Array[Int] = null;
	var inbound_counts: Array[Int] = null;
	var outbound_displs: Array[Int] = null;
	var outbound_counts: Array[Int] = null;
	
	
	var sTime = 0l;
	var eTime = 0l;

	def init(){
		nvertices = args(0).toInt;
		
		comm = new SparseComm(args);
		size = comm.size;
		
		nv_per_proc = (nvertices + size -1)/size;
		
		if(comm.rank == 0){
			println("size: %d, nvertices: %d, nv_per_proc: %d".format(size, nvertices, nv_per_proc));
		}
		
		if(comm.rank == 0){
			println("initialize model parameters ...");
		}
		initModel();
		
		if(comm.rank == 0){
			println("vertices assignment ...");
		}
		partition();
		
		if(comm.rank == 0){
			println("reduce config ...");
		}
		reduceConfig();
		
		if(comm.rank == 0){
			println("initialize vector ...");
		}
		vector = Array.fill[Float](nv_per_proc)(0f);
	
	}
	
	def partition(){
		
		inbound_displs = Array.fill[Int](size+1)(0);
		inbound_counts = Array.fill[Int](size)(0);
		outbound_displs = Array.fill[Int](size+1)(0);
		outbound_counts = Array.fill[Int](size)(0);
		
		var lpid: Int = -1;
		var pid: Int = 0;
		
		for( i <- 0 to inboundIndices.length - 1 ){
			pid = inboundIndices(i)/nv_per_proc;
			if( lpid != pid ){
				for( j <- lpid+1 to pid ){
					inbound_displs(j) = i;
				}
			}
			lpid = pid;
		}
		
		for( i <- lpid+1 to size){
			inbound_displs(i) = inboundIndices.length;
		}
		
		for( i <- 0 to size-1){
			inbound_counts(i) = inbound_displs(i+1) - inbound_displs(i);
		}
		
		
		lpid = -1;
		pid = 0;

		for( i <- 0 to outboundIndices.length - 1 ){
			pid = outboundIndices(i)/nv_per_proc;
			if( lpid != pid ){
				for( j <- lpid+1 to pid ){
					outbound_displs(j) = i;
				}
			}
			lpid = pid;
		}
		
		for( i <- lpid+1 to size){
			outbound_displs(i) = outboundIndices.length;
		}
		
		for( i <- 0 to size-1){
			outbound_counts(i) = outbound_displs(i+1) - outbound_displs(i);
		}
	}
	
	def initModel(){
		//load data

		//val row: IMat = load("/home/ubuntu/data/TwitterGraph/tgraph%d-rl.mat".format(comm.rank), "row");
		//val col: IMat = load("/home/ubuntu/data/TwitterGraph/tgraph%d-rl.mat".format(comm.rank), "col");
		
		//val row: IMat = load("/home/ubuntu/data/TwitterGraph/tgraph.mat", "row");
		//val col: IMat = load("/home/ubuntu/data/TwitterGraph/tgraph.mat", "col");
		

		val ndata: Int = 32/size;
		
		var rt:IMat = null;
		var ct:IMat = null;
		var row:IMat = null;
		var col:IMat = null;

		for(i <- 0 to ndata-1){

			if(i==0){
				row = load("/home/ubuntu/data/TwitterGraph/tgraph%d-rl.mat".format(comm.rank), "row");
				col = load("/home/ubuntu/data/TwitterGraph/tgraph%d-rl.mat".format(comm.rank), "col");
			}else{
				
				rt = load("/home/ubuntu/data/TwitterGraph/tgraph%d-rl.mat".format(comm.rank+i*size), "row");
				ct = load("/home/ubuntu/data/TwitterGraph/tgraph%d-rl.mat".format(comm.rank+i*size), "col");
				row = row on rt;
				col = col on ct;

			}
		
		}

		val (ur, ir, jr) = unique(row);
		val (uc, ic, jc) = unique(col);
				
		nr = length(ur);
		nc = length(uc);

		println("rank: %d, nr: %d, nc: %d".format(comm.rank, nr, nc));
		
		// map from vertex indices to compact indices, for internal use only, in and out flipped		
		inMap = Map[Int, Int]();
		outMap = Map[Int, Int]();
		
		var id: Int = 0;
		
		for( i <- 0 to nr-1 ){
			id = ur(i);
			inMap += (id -> i);
		}
		
		for( i <- 0 to nc-1 ){
			id = uc(i);
			outMap += (id -> i);
		}
		
		inboundIndices = uc.data;
		outboundIndices = ur.data;

		inboundValues = Array.fill[Float](nc)(1f/nvertices);

		val ne = length(row);
		
		var rmap: Int = 0;
		var cmap: Int = 0;
		var v: Int = 0;
		
		for(i <- 0 to ne-1){

			v = row(i);
			rmap = inMap(v);
			row(i) = rmap;
			
			v = col(i);
			cmap = outMap(v);
			col(i) = cmap;
		}
		
		edgeMat = sparse(row, col, ones(ne, 1));

	}
	
	def update(){
		
		in = FMat(nc, 1, inboundValues);
		out = edgeMat * in;
		outboundValues = out.data;
		
	}
	
	def reduceConfig(){
	
		comm.scatterConfig(outboundIndices, outbound_counts, outbound_displs);
		comm.gatherConfig(inboundIndices, inbound_counts, inbound_displs);
		
	}
	
	def reduce(){
	
		comm.scatter(outboundValues, outbound_counts, outbound_displs, vector,  nv_per_proc);
		comm.gather(vector, inboundValues, inbound_counts, nv_per_proc);
		
	}
	
	def run(){
		
		for(i <- 1 to 10){
		if(comm.rank == 0){
			println("model update ...");
		}
		
		flip;
		update();
		val fu = gflop;
		
		if(comm.rank == 0){
			println("compute (gflops, time)" + fu);
		}
		
		if(comm.rank == 0){
			println("communicate model ...");
		}
		
		//flip;
		sTime = System.nanoTime;
		reduce();
		eTime = System.nanoTime;
		//val fr = gflop;
		

		//println("processor %d old comm time: %f s)".format(comm.rank, fr._2));
		println("processor %d comm time: %f s)".format(comm.rank, (eTime-sTime)/1000000000f));
		//println("processor %d sendrecv time: %f s".format(comm.rank, comm.getTime));
		println("processor %d throughput: %f GB/sec".format(comm.rank, comm.getThroughput));
		
		vector = Array.fill[Float](nv_per_proc)(0f);		
		}
		
		
	}
	
	def terminate(){
		
		if(comm.rank == 0){
			println("terminating ...");
		}
		comm.terminate();
	}

}

object Model{
	def main(args: Array[String]){
		val model: Model = new Model(args);
		model.init();
		model.run();
		model.terminate();
	}
}
