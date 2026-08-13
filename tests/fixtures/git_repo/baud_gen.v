module baud_gen #(
    parameter CLK_FREQ = 50_000_000,
    parameter BAUD_RATE = 9600
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] baud_div,
    output reg         baud_tick
);

    localparam DIV_DEFAULT = CLK_FREQ / (BAUD_RATE * 16);

    reg [15:0] counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter   <= 16'd0;
            baud_tick <= 1'b0;
        end else begin
            if (counter >= baud_div) begin
                counter   <= 16'd0;
                baud_tick <= 1'b1;
            end else begin
                counter   <= counter + 1'b1;
                baud_tick <= 1'b0;
            end
        end
    end

endmodule
