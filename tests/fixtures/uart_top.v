module uart_top #(
    parameter CLK_FREQ  = 50_000_000,
    parameter BAUD_RATE = 115200,
    parameter DATA_BITS = 8
)(
    input  wire                  clk,
    input  wire                  rst_n,
    // TX interface
    input  wire [DATA_BITS-1:0]  tx_data,
    input  wire                  tx_valid,
    output wire                  tx_out,
    output wire                  tx_busy,
    // RX interface
    input  wire                  rx_in,
    output wire [DATA_BITS-1:0]  rx_data,
    output wire                  rx_valid,
    // Config
    input  wire [15:0]           baud_div
);

    wire baud_tick;

    baud_gen #(
        .CLK_FREQ  (CLK_FREQ),
        .BAUD_RATE (BAUD_RATE)
    ) u_baud_gen (
        .clk       (clk),
        .rst_n     (rst_n),
        .baud_div  (baud_div),
        .baud_tick (baud_tick)
    );

    uart_tx #(
        .DATA_BITS (DATA_BITS)
    ) u_uart_tx (
        .clk       (clk),
        .rst_n     (rst_n),
        .baud_tick (baud_tick),
        .tx_data   (tx_data),
        .tx_valid  (tx_valid),
        .tx_out    (tx_out),
        .tx_busy   (tx_busy)
    );

    uart_rx #(
        .DATA_BITS (DATA_BITS)
    ) u_uart_rx (
        .clk       (clk),
        .rst_n     (rst_n),
        .baud_tick (baud_tick),
        .rx_in     (rx_in),
        .rx_data   (rx_data),
        .rx_valid  (rx_valid)
    );

endmodule
