`timescale 1ns/1ps
module tb_uart;
 localparam N=16;
 reg clk=0;always #5 clk=~clk;
 reg rst=1,valid=0;reg[7:0] data=0;wire tx,ready,busy,done;
 uart_tx_clean #(.N(N)) u_tx(clk,rst,valid,data,tx,ready,busy,done);
 reg rx=1;wire rv;wire[7:0] rd;
 uart_rx_clean #(.N(N)) u_rx(clk,rst,rx,rv,rd);
 integer i,j,received=0;reg[7:0] expected_rx;
 always @(posedge clk)begin
  #1;if(rv)begin
   if(rd!==expected_rx)$fatal(1,"RX %h expected %h",rd,expected_rx);
   received=received+1;
  end
 end
 task send_rx(input[7:0] value,input bit bad_stop);
  begin
   expected_rx=value;@(negedge clk);rx=0;
   repeat(N)@(negedge clk);
   for(j=0;j<8;j=j+1)begin rx=value[j];repeat(N)@(negedge clk);end
   rx=bad_stop?0:1;repeat(N)@(negedge clk);
   rx=1;repeat(N)@(negedge clk);
  end
 endtask
 reg[9:0] frame;
 initial begin
  repeat(4)@(negedge clk);rst=0;
  for(i=0;i<256;i=i+1)begin
   wait(ready);@(negedge clk);data=i;valid=1;
   @(posedge clk);#1;frame={1'b1,data,1'b0};
   @(negedge clk);valid=0;data=~data;
   repeat(N/2-1)@(negedge clk);
   for(j=0;j<10;j=j+1)begin
    if(tx!==frame[j])$fatal(1,"TX byte %d bit %d",i,j);
    repeat(N)@(negedge clk);
   end
  end
  for(i=0;i<256;i=i+1)send_rx(i,0);
  if(received!=256)$fatal(1,"RX count");
  // Short start glitch must not produce an event.
  @(negedge clk);rx=0;repeat(2)@(negedge clk);rx=1;
  repeat(12*N)@(negedge clk);
  if(received!=256)$fatal(1,"glitch accepted");
  send_rx(8'hA5,1);
  if(received!=256)$fatal(1,"bad stop accepted");
  send_rx(8'h3C,0);
  if(received!=257)$fatal(1,"recovery");
  $display("PASS: TX 256 frames, RX 256 frames, short glitch rejected, bad stop rejected, valid recovery");$finish;
 end
 initial begin #2000000;$fatal(1,"timeout");end
endmodule
