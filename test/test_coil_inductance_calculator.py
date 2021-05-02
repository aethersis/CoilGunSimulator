from src.coil_inductance import calc_multi_layer_for_known_layer_count


def test_valid_result_against_coil32_net():
    # see https://coil32.net/online-calculators/multilayer-coil-calculator.html
    result = calc_multi_layer_for_known_layer_count(5, 10, 28, 1, 1.09)
    assert result.DC_resistance_ohms == 0.08
    assert round(result.inductance_microhenries) == 31
    assert result.required_wire_length_meters == 3.622
    assert result.total_number_of_turns == 101
    assert result.winding_thickness_millimeters == 5.45


test_valid_result_against_coil32_net()
