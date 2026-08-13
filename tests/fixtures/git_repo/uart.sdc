# Clock definitions
create_clock -period 10.0 -name clk_core [get_ports clk]
create_generated_clock -name clk_baud -source [get_ports clk] -divide_by 434 [get_pins u_baud_gen/baud_tick]

# Input delays
set_input_delay -clock clk_core -max 5.0 [get_ports rx_in]
set_input_delay -clock clk_core -min 1.0 [get_ports rx_in]
set_input_delay -clock clk_core -max 3.0 [get_ports {tx_data[*]}]
set_input_delay -clock clk_core -max 3.0 [get_ports tx_valid]
set_input_delay -clock clk_core -max 2.0 [get_ports {baud_div[*]}]

# Output delays
set_output_delay -clock clk_core -max 4.0 [get_ports tx_out]
set_output_delay -clock clk_core -max 4.0 [get_ports tx_busy]
set_output_delay -clock clk_core -max 5.0 [get_ports {rx_data[*]}]
set_output_delay -clock clk_core -max 3.0 [get_ports rx_valid]

# False paths
set_false_path -from [get_ports rst_n]
set_false_path -from [get_clocks clk_core] -to [get_clocks clk_baud]

# Multicycle paths
set_multicycle_path 2 -setup -from [get_pins u_baud_gen/counter_reg[*]] -to [get_pins u_uart_tx/shift_reg[*]]

# Max delay
set_max_delay 10.0 -from [get_ports rx_in] -to [get_ports {rx_data[*]}]

# Clock uncertainty
set_clock_uncertainty 0.5 [get_clocks clk_core]

# Load and driving cell
set_load 0.05 [get_ports tx_out]
set_driving_cell -lib_cell BUF_X2 [get_ports clk]
