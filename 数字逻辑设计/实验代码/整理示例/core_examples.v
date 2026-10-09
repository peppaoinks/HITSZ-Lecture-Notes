`timescale 1ns/1ps
`default_nettype none
module decoder_38(input wire [2:0] en,data_in,
                   output wire [7:0] data_out);
 wire active = en[2] & ~en[1] & ~en[0];
 assign data_out = active ? ~(8'b1 << data_in) : 8'hFF;
endmodule
module mux_clean(input wire en,mux_sel,
 input wire [3:0] input_a,input_b,output reg [3:0] output_c);
 always @* begin
  if (!en) output_c = 4'hF;
  else if (!mux_sel) output_c = input_a + input_b;
  else output_c = input_a - input_b;
 end
endmodule
module dffe(input wire clk,clrn,wen,d,output reg q);
 always @(posedge clk or negedge clrn)
  if(!clrn) q <= 1'b0;
  else if(!wen) q <= d;
endmodule
module reg8(input wire clk,clrn,wen,
 input wire [7:0] d, output wire [7:0] q);
 genvar b;
 generate for(b=0;b<8;b=b+1) begin: bits
  dffe u_ff(.clk(clk),.clrn(clrn),.wen(wen),.d(d[b]),.q(q[b]));
 end endgenerate
endmodule
module decoder_write_n(input wire wen,input wire [2:0] wsel,
 output wire [7:0] we_n);
 assign we_n = wen ? 8'hFF : ~(8'b1 << wsel);
endmodule
module read_mux8(input wire [2:0] rsel,
 input wire [63:0] values,output wire [7:0] q);
 assign q = values[8*rsel +: 8];
endmodule
module reg8file_struct(input wire clk,clrn,wen,
 input wire [7:0] d,input wire [2:0] wsel,rsel,
 output wire [7:0] q);
 wire [7:0] we_n;
 wire [63:0] values;
 decoder_write_n u_dec(.wen(wen),.wsel(wsel),.we_n(we_n));
 genvar j;
 generate for(j=0;j<8;j=j+1) begin: words
  reg8 u_reg(.clk(clk),.clrn(clrn),.wen(we_n[j]),.d(d),
             .q(values[8*j +: 8]));
 end endgenerate
 read_mux8 u_read(.rsel(rsel),.values(values),.q(q));
endmodule
module reg8file_clean(input wire clk,clr,en,input wire [7:0] d,
 input wire [2:0] wsel,rsel,output wire [7:0] q);
 reg [7:0] mem [0:7]; integer i;
 always @(posedge clk or posedge clr)
  if(clr) begin for(i=0;i<8;i=i+1) mem[i]<=0; end
  else if(en) mem[wsel] <= d;
 assign q = mem[rsel];
endmodule
module tick_enable #(parameter integer CYCLES=100000)(
 input wire clk,rst,en,output wire tick);
 localparam integer W=(CYCLES<=1)?1:$clog2(CYCLES);
 reg [W-1:0] count;
 assign tick = en && (count==CYCLES-1) && !rst;
 always @(posedge clk or posedge rst)
  if(rst) count <= 0;
  else if(en) begin
   if(tick) count <= 0;
   else count <= count+1'b1;
  end
endmodule
module button_clean #(parameter integer STABLE_CYCLES=2000000)(
 input wire clk,rst,button,output reg stable,output wire press);
 (* ASYNC_REG="TRUE" *) reg sync1,sync2;
 reg delayed;
 localparam integer W=(STABLE_CYCLES<=1)?1:$clog2(STABLE_CYCLES);
 reg [W-1:0] count;
 always @(posedge clk or posedge rst) begin
  if(rst) begin sync1<=0;sync2<=0;end
  else begin sync1<=button;sync2<=sync1;end
 end
 always @(posedge clk or posedge rst) begin
  if(rst) begin count<=0;stable<=0;delayed<=0;end
  else begin
   delayed <= stable;
   if(sync2==stable) count<=0;
   else if(count==STABLE_CYCLES-1) begin
    stable<=sync2;count<=0;
   end else count<=count+1'b1;
  end
 end
 assign press=stable & ~delayed;
endmodule
module gray_counter(input wire clk_i,rst_n_i,en_i,
 output wire [3:0] gray_o);
 reg [3:0] binary;
 always @(posedge clk_i or negedge rst_n_i)
  if(!rst_n_i) binary<=0;
  else if(en_i) binary<=binary+1'b1;
 assign gray_o=binary ^ (binary>>1);
endmodule
module decimal_100(input wire clk,rst,inc,
 output reg [3:0] tens,ones);
 always @(posedge clk or posedge rst) begin
  if(rst) begin tens<=0;ones<=0;end
  else if(inc) begin
   if(ones==9) begin ones<=0;tens<=(tens==9)?0:tens+1'b1;end
   else ones<=ones+1'b1;
  end
 end
endmodule
module seq01011_word #(parameter integer MOORE=0)(
 input wire clk,rst,start,input wire [7:0] data_in,
 output reg busy,detect);
 reg [7:0] data_buf;
 reg [3:0] left;
 reg [2:0] state;
 reg [2:0] next_state;
 wire bit_in = data_buf[7];
 wire mealy_hit = (state==4) && bit_in;
 always @* begin
  case(state)
   0:next_state=bit_in?0:1;
   1:next_state=bit_in?2:1;
   2:next_state=bit_in?0:3;
   3:next_state=bit_in?4:1;
   4:next_state=bit_in?(MOORE?5:0):3;
   5:next_state=bit_in?0:1;
   default:next_state=0;
  endcase
 end
 always @(posedge clk or posedge rst) begin
  if(rst) begin
   busy<=0;detect<=0;state<=0;left<=0;data_buf<=0;
  end else if(start) begin
   busy<=1;detect<=0;state<=0;left<=8;data_buf<=data_in;
  end else if(busy) begin
   if(MOORE && state==5) detect<=1;
   if(left!=0) begin
    if(!MOORE && mealy_hit) detect<=1;
    state<=next_state;
    data_buf<={data_buf[6:0],1'b0};
    left<=left-1'b1;
    // Leave one drain cycle for the Moore hit state.
    if(left==1 && !MOORE) busy<=0;
   end else begin busy<=0;state<=0;end
  end
 end
endmodule
`default_nettype wire
