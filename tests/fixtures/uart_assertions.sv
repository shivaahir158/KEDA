// SVA assertions for UART design
module uart_assertions (
    input wire       clk,
    input wire       rst_n,
    input wire       tx_valid,
    input wire       tx_busy,
    input wire       tx_out,
    input wire       rx_in,
    input wire       rx_valid,
    input wire [7:0] rx_data,
    input wire       baud_tick
);

    // A1: tx_busy must assert within 2 cycles of tx_valid
    property p_tx_busy_assert;
        @(posedge clk) disable iff (!rst_n)
        tx_valid && !tx_busy |-> ##[1:2] tx_busy;
    endproperty
    assert property (p_tx_busy_assert)
        else $error("TX busy did not assert after tx_valid");

    // A2: tx_out should be high (idle) when not busy
    a_tx_idle: assert property (
        @(posedge clk) disable iff (!rst_n)
        !tx_busy |-> tx_out
    ) else $error("TX line not idle when not busy");

    // A3: rx_valid should be a single-cycle pulse
    a_rx_valid_pulse: assert property (
        @(posedge clk) disable iff (!rst_n)
        rx_valid |=> !rx_valid
    ) else $error("rx_valid was not a single-cycle pulse");

    // A4: baud_tick should not stay high for more than 1 cycle
    a_baud_tick_pulse: assert property (
        @(posedge clk) disable iff (!rst_n)
        baud_tick |=> !baud_tick
    ) else $error("baud_tick was not a single-cycle pulse");

    // A5: Cover successful TX-to-RX loopback
    c_loopback: cover property (
        @(posedge clk)
        tx_valid ##[1:$] rx_valid
    );

    // A6: Assume no simultaneous tx_valid during tx_busy
    a_no_tx_during_busy: assume property (
        @(posedge clk) disable iff (!rst_n)
        tx_busy |-> !tx_valid
    );

    // A7: After reset, tx_busy should be low
    a_reset_tx_idle: assert property (
        @(posedge clk)
        !rst_n |=> !tx_busy
    );

endmodule

// Inline assertion inside a module (common in real RTL)
module baud_gen_with_asserts #(
    parameter CLK_FREQ = 50_000_000,
    parameter BAUD_RATE = 115200
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] baud_div,
    output reg         baud_tick
);
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

    // Inline SVA
    a_counter_no_overflow: assert property (
        @(posedge clk) disable iff (!rst_n)
        counter <= baud_div
    ) else $error("Counter exceeded baud_div");

    a_tick_one_hot: assert property (
        @(posedge clk) disable iff (!rst_n)
        baud_tick |=> !baud_tick
    );

endmodule
