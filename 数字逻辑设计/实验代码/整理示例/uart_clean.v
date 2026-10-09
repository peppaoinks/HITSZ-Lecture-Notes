`timescale 1ns/1ps
`default_nettype none
// Integer clocks per serial bit, with N >= 4.
module uart_tx_clean #(parameter integer N=10416)(
 input wire clk,rst,valid,input wire [7:0] data,
 output reg tx,output wire ready,busy,output reg done);
 localparam [1:0] IDLE=0,START=1,DATA=2,STOP=3;
 localparam integer W=(N<=1)?1:$clog2(N);
 reg [1:0] state,next_state;
 reg [W-1:0] count;
 reg [2:0] bit_index;
 reg [7:0] buffer;
 wire tick=(count==N-1);
 assign ready=(state==IDLE) && !rst;
 assign busy=(state!=IDLE);
 always @* begin
  next_state=state;
  case(state)
   IDLE:if(valid) next_state=START;
   START:if(tick) next_state=DATA;
   DATA:if(tick && bit_index==7) next_state=STOP;
   STOP:if(tick) next_state=IDLE;
  endcase
 end
 always @(posedge clk or posedge rst)
  if(rst) state<=IDLE;
  else state<=next_state;
 always @(posedge clk or posedge rst) begin
  if(rst) begin count<=0;bit_index<=0;buffer<=0;tx<=1;done<=0;end
  else begin
   done<=0;
   if(state==IDLE) begin
    count<=0;bit_index<=0;tx<=1;
    if(valid) begin buffer<=data;tx<=0;end
   end else if(tick) begin
    count<=0;
    case(state)
     START:begin bit_index<=0;tx<=buffer[0];end
     DATA:begin
      if(bit_index==7) tx<=1;
      else begin bit_index<=bit_index+1'b1;tx<=buffer[bit_index+1];end
     end
     STOP:begin tx<=1;done<=1;end
    endcase
   end else count<=count+1'b1;
  end
 end
endmodule
module uart_rx_clean #(parameter integer N=10416)(
 input wire clk,rst,rx,output reg valid,output reg [7:0] data);
 localparam [1:0] IDLE=0,START=1,RECV=2,STOP=3;
 localparam integer W=(N<=1)?1:$clog2(N);
 (* ASYNC_REG="TRUE" *) reg sync1,sync2;
 reg [1:0] state,next_state;
 reg [W-1:0] count;
 reg [2:0] bit_index;
 reg [7:0] buffer;
 wire half=(count==N/2-1);
 wire tick=(count==N-1);
 always @(posedge clk or posedge rst)
  if(rst) begin sync1<=1;sync2<=1;end
  else begin sync1<=rx;sync2<=sync1;end
 always @* begin
  next_state=state;
  case(state)
   IDLE:if(!sync2) next_state=START;
   START:if(half) next_state=sync2?IDLE:RECV;
   RECV:if(tick && bit_index==7) next_state=STOP;
   STOP:if(tick) next_state=IDLE;
  endcase
 end
 always @(posedge clk or posedge rst)
  if(rst) state<=IDLE;
  else state<=next_state;
 always @(posedge clk or posedge rst) begin
  if(rst) begin count<=0;bit_index<=0;buffer<=0;valid<=0;data<=0;end
  else begin
   valid<=0;
   if(state==IDLE) begin count<=0;bit_index<=0;end
   else if(state==START && half) count<=0;
   else if((state==RECV || state==STOP) && tick) begin
    count<=0;
    if(state==RECV) begin
     buffer[bit_index]<=sync2;
     if(bit_index!=7) bit_index<=bit_index+1'b1;
    end else if(sync2) begin data<=buffer;valid<=1;end
   end else count<=count+1'b1;
  end
 end
endmodule
`default_nettype wire
