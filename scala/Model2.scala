import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._


class Model2(args: Array[String]){

	var inboundIndices: Array[Int] = null;
	var inboundValues: Array[Float] = null;
	var outboundIndices: Array[Int] = null;
	var outboundValues: Array[Float] = null;
	
	var vector: Array[Float] = null;
	
	
	var dim: Int = -1;
	var size: Int = -1;
	var dim_per_proc: Int = -1;

	def init(){
		
		initModel();
	
	}
	
	def initModel(){
		//load data
		var a = rand(4, 1);
		var b = rand(1, 4);

		var c = b*a;

		println(a);
	
		println("loading data ... \n");
		
		val row: IMat = load("/home/ec2-user/data/TwitterGraph/tgraph0.mat", "row");
		val col: IMat = load("/home/ec2-user/data/TwitterGraph/tgraph0.mat", "col");
		

		inboundIndices = col.data;
		println("inboundIndices: " + inboundIndices.length);
		outboundIndices = row.data;
		println("outboundIndices: " + outboundIndices.length);
		
	}

}

object Model2{
	def main(args: Array[String]){
		val model: Model2 = new Model2(args);
		model.init();
	}
}
