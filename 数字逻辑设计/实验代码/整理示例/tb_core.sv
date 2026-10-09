`timescale 1ns/1ps
module tb_core;
 reg clk=0;always #5 clk=~clk;
 reg rst=1,en=0,sel=0;reg [3:0] a=0,b=0;wire[3:0] y;
 mux_clean m(en,sel,a,b,y);
 reg[2:0] den=0,addr=0;wire[7:0] dy;
 decoder_38 dec(den,addr,dy);
 reg[7:0] d=0;reg[2:0] ws=0,rs=0;wire[7:0] q,qs;
 reg8file_clean rf(clk,rst,en,d,ws,rs,q);
 reg8file_struct rfs(clk,~rst,~en,d,ws,rs,qs);
 reg start=0;reg[7:0] word=0;wire busy0,busy1,hit0,hit1;
 seq01011_word #(.MOORE(0)) sq0(clk,rst,start,word,busy0,hit0);
 seq01011_word #(.MOORE(1)) sq1(clk,rst,start,word,busy1,hit1);
 wire[3:0] gray;
 gray_counter gc(clk,~rst,en,gray);
 reg button=0;wire stable,press;
 button_clean #(.STABLE_CYCLES(4)) bc(clk,rst,button,stable,press);
 integer presses=0;
 always @(posedge clk) if(press) presses=presses+1;
 wire tick;
 tick_enable #(.CYCLES(4)) te(clk,rst,en,tick);
 reg inc=0;wire[3:0] tens,ones;
 decimal_100 dc(clk,rst,inc,tens,ones);
 integer e,s,x,z,i,k,ticks;reg[3:0] expected;reg found;
 reg[3:0] oldgray,delta;
 task reset_all;
  begin @(negedge clk);rst=1;en=0;inc=0;start=0;
   @(negedge clk);rst=0; end
 endtask
 initial begin
  en=1;#1;
  for(e=0;e<2;e=e+1)for(s=0;s<2;s=s+1)
   for(x=0;x<16;x=x+1)for(z=0;z<16;z=z+1)begin
    en=e;sel=s;a=x;b=z;#1;
    expected=!e?15:(!s?x+z:x-z);
    if(y!==expected)$fatal(1,"mux");
   end
  for(e=0;e<8;e=e+1)for(x=0;x<8;x=x+1)begin
   den=e;addr=x;#1;
   if(dy!==((e==4)?~(8'b1<<x):8'hFF))$fatal(1,"decoder");
  end
  reset_all();
  for(i=0;i<8;i=i+1)begin
   @(negedge clk);en=1;ws=i;d=8'hA0+i;
   @(posedge clk);#1;
  end
  @(negedge clk);en=0;d=8'hFF;
  for(i=0;i<8;i=i+1)begin
   rs=i;#1;if(q!==8'hA0+i || qs!==q)$fatal(1,"regfile");
  end
  repeat(3)@(posedge clk);#1;
  if(q!==8'hA7 || qs!==q)$fatal(1,"hold");
  #2;rst=1;#1;if(q!==0 || qs!==0)$fatal(1,"async reset");
  reset_all();
  for(i=0;i<256;i=i+1)begin
   word=i;found=0;
   for(k=0;k<=3;k=k+1)if(((i>>k)&31)==11)found=1;
   @(negedge clk);start=1;
   @(negedge clk);start=0;
   repeat(11)@(negedge clk);
   if(hit0!==found || hit1!==found)$fatal(1,"sequence %h",word);
  end
  reset_all();en=1;oldgray=0;
  repeat(32)begin
   @(posedge clk);#1;delta=gray^oldgray;
   if(delta==0 || (delta & (delta-1))!=0)$fatal(1,"gray");
   oldgray=gray;
  end
  @(negedge clk);en=0;
  repeat(3)@(posedge clk);#1;
  if(gray!==oldgray)$fatal(1,"gray hold");
  reset_all();ticks=0;en=1;
  repeat(20)begin @(posedge clk);if(tick)ticks=ticks+1;end
  if(ticks!=5)$fatal(1,"tick period %0d",ticks);
  @(negedge clk);en=0;
  reset_all();button=1;
  repeat(2)@(negedge clk);button=0;
  repeat(10)@(negedge clk);
  if(stable!==0 || presses!=0)$fatal(1,"bounce");
  button=1;repeat(15)@(negedge clk);
  if(stable!==1 || presses!=1)$fatal(1,"press once");
  repeat(15)@(negedge clk);if(presses!=1)$fatal(1,"long hold");
  button=0;repeat(15)@(negedge clk);
  button=1;repeat(15)@(negedge clk);
  if(presses!=2)$fatal(1,"second press");
  reset_all();inc=1;
  for(i=1;i<=256;i=i+1)begin
   @(posedge clk);#1;
   if(tens!==((i%100)/10) || ones!==(i%10))$fatal(1,"decimal %0d",i);
  end
  $display("PASS: mux 1024, decoder 64, regfiles 8 addresses, sequence 256 words x 2, gray 32 transitions, tick, debounce, decimal 256");
  $finish;
 end
endmodule
