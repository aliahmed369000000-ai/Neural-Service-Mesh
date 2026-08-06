`timescale 1ns/1ps
module tb_matmul_pe;
    reg clk = 0, rst_n = 0, enable = 0;
    reg [15:0] a = 0, b = 0;
    wire [31:0] acc;
    matmul_pe dut(.clk(clk), .rst_n(rst_n), .enable(enable), .a_q88(a), .b_q88(b), .acc(acc));
    always #5 clk = ~clk;
    initial begin
        #12 rst_n = 1;
        enable = 1;
        a = 16'sd256; // 1.0 in Q8.8
        b = 16'sd256;
        #20;
        $display("NSM ASIC PE acc=%0d (expect ~65536 for 1*1 in Q8.8 product scale)", acc);
        $finish;
    end
endmodule
