use GPUIterator;
use Time;
use Random;

config const n = 4096: int;
config const CPU_percentage = 0: int;
config param useGPU = false;

var A: [1..n*n] int(32);
var B: [1..n*n] int(32);
var C: [1..n*n] int(32);
var run_time: real(64);

var range1 = new owned RandomStream(eltType = int(32));
var range2 = new owned RandomStream(eltType = int(32));

extern proc mulCUDA(A: [] int(32), B: [] int(32), C: [] int(32), lo: int, hi: int, N: int) where useGPU == true;

proc GPUCallBack(lo: int, hi: int, N: int) {
if (hi-lo+1 != N) {
  writeln('exit');
  exit();
}
if (useGPU == false) {
    writeln('no GPU found');
}else{
  mulCUDA(A, B, C, lo, hi, N);
}

}

for i in 1..n*n do{
  A(i) = range1.getNext(min = 0, max = 10);
  B(i) = range2.getNext(min = 0, max = 10);
}

const start = getCurrentTime(TimeUnits.milliseconds);

forall e in GPU(1..n, GPUCallBack, CPU_percentage) {
//Here  goes a CPU  function  that is the  equivalent  tothe  kernel  function
}

run_time = getCurrentTime(TimeUnits.milliseconds) - start;


write('\n');
write("Time to execute the Matrix Multiplication: ",run_time, ' milliseconds\n');
