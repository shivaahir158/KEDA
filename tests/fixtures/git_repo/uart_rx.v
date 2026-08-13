module uart_rx #(
    parameter DATA_BITS = 16
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  baud_tick,
    input  wire                  rx_in,
    output reg  [DATA_BITS-1:0]  rx_data,
    output reg                   rx_valid
);

    localparam IDLE  = 2'b00;
    localparam START = 2'b01;
    localparam DATA  = 2'b10;
    localparam STOP  = 2'b11;

    reg [1:0] state;
    reg [2:0] bit_idx;
    reg [DATA_BITS-1:0] shift_reg;
    reg [3:0] sample_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            rx_data    <= {DATA_BITS{1'b0}};
            rx_valid   <= 1'b0;
            bit_idx    <= 3'd0;
            shift_reg  <= {DATA_BITS{1'b0}};
            sample_cnt <= 4'd0;
        end else begin
            rx_valid <= 1'b0;
            case (state)
                IDLE: begin
                    if (!rx_in) begin
                        state      <= START;
                        sample_cnt <= 4'd0;
                    end
                end
                START: begin
                    if (baud_tick) begin
                        if (sample_cnt == 4'd7) begin
                            if (!rx_in) begin
                                state      <= DATA;
                                bit_idx    <= 3'd0;
                                sample_cnt <= 4'd0;
                            end else begin
                                state <= IDLE;
                            end
                        end else begin
                            sample_cnt <= sample_cnt + 1'b1;
                        end
                    end
                end
                DATA: begin
                    if (baud_tick) begin
                        if (sample_cnt == 4'd15) begin
                            shift_reg <= {rx_in, shift_reg[DATA_BITS-1:1]};
                            sample_cnt <= 4'd0;
                            if (bit_idx == DATA_BITS - 1)
                                state <= STOP;
                            else
                                bit_idx <= bit_idx + 1'b1;
                        end else begin
                            sample_cnt <= sample_cnt + 1'b1;
                        end
                    end
                end
                STOP: begin
                    if (baud_tick) begin
                        if (sample_cnt == 4'd15) begin
                            if (rx_in) begin
                                rx_data  <= shift_reg;
                                rx_valid <= 1'b1;
                            end
                            state <= IDLE;
                        end else begin
                            sample_cnt <= sample_cnt + 1'b1;
                        end
                    end
                end
            endcase
        end
    end

endmodule
