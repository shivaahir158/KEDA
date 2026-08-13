module uart_tx #(
    parameter DATA_BITS = 16
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  baud_tick,
    input  wire [DATA_BITS-1:0]  tx_data,
    input  wire                  tx_valid,
    output reg                   tx_out,
    output reg                   tx_busy
);

    localparam IDLE  = 2'b00;
    localparam START = 2'b01;
    localparam DATA  = 2'b10;
    localparam STOP  = 2'b11;

    reg [1:0] state;
    reg [2:0] bit_idx;
    reg [DATA_BITS-1:0] shift_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            tx_out    <= 1'b1;
            tx_busy   <= 1'b0;
            bit_idx   <= 3'd0;
            shift_reg <= {DATA_BITS{1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    tx_out <= 1'b1;
                    if (tx_valid) begin
                        state     <= START;
                        shift_reg <= tx_data;
                        tx_busy   <= 1'b1;
                    end
                end
                START: begin
                    if (baud_tick) begin
                        tx_out  <= 1'b0;
                        state   <= DATA;
                        bit_idx <= 3'd0;
                    end
                end
                DATA: begin
                    if (baud_tick) begin
                        tx_out <= shift_reg[0];
                        shift_reg <= shift_reg >> 1;
                        if (bit_idx == DATA_BITS - 1)
                            state <= STOP;
                        else
                            bit_idx <= bit_idx + 1'b1;
                    end
                end
                STOP: begin
                    if (baud_tick) begin
                        tx_out  <= 1'b1;
                        tx_busy <= 1'b0;
                        state   <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule
