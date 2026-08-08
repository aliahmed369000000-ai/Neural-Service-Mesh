// NSM Processing Element — MAC primitive (design sketch)
// y += a * b  (fixed-point Q8.8 example)
module matmul_pe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire [15:0] a_q88,
    input  wire [15:0] b_q88,
    output reg  [31:0] acc
);
    wire signed [31:0] prod = $signed(a_q88) * $signed(b_q88);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 32'sd0;
        else if (enable)
            acc <= acc + prod;
    end
endmodule
