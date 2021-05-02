#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:15:35 2012
Updated on Mon Jul 15 19:54:35 2013
@author: Valery Kustarev kustva@gmail.com
Extended by Maksym Walczak
"""
import math
from dataclasses import dataclass


def elliptic_integral(c):
    a = 1
    b = math.sqrt(1 - pow(c, 2))
    E = 1 - pow(c, 2) / 2
    i = 1
    while abs(a - b) > 1e-15:
        a1 = (a + b) / 2
        b1 = math.sqrt(a * b)
        E = E - i * pow(((a - b) / 2), 2)
        i = 2 * i
        a = a1
        b = b1
    Fk = math.pi / (2 * a)
    Ek = E * Fk
    return [Fk, Ek]


def mutual_inductance(r1, r2, x, g):
    l = math.sqrt(pow((r2 - r1), 2) + pow(x, 2))
    c = 2 * math.sqrt(r1 * r2) / math.sqrt(pow((r1 + r2), 2) + pow(l - g, 2))
    e = elliptic_integral(c)
    Result = (
        -0.004 * math.pi * math.sqrt(r1 * r2) * ((c - 2 / c) * e[0] + (2 / c) * e[1])
    )
    return Result


@dataclass
class ResultingCoilParameters:
    inductance_microhenries: float
    total_number_of_turns: float
    winding_thickness_millimeters: float
    DC_resistance_ohms: float
    required_wire_length_meters: float


def calc_multi_layer_for_known_layer_count(
    layers_count,
    inner_diameter,
    length,
    wire_without_insulation_thickness,
    wire_with_insulation_thickness,
) -> ResultingCoilParameters:
    """
    Calculates parameters of a multi layer coil given the number of layers, inner diameter, length, and wire thickness
    :param layers_count: number of layers in the coil (radial turns)
    :param inner_diameter: inner diameter of the coil (aka. former diameter)
    :param length: length of the coil in mm
    :param wire_without_insulation_thickness: thickness of the coil wire without insulation in mm
    :param wire_with_insulation_thickness:  thickness of the coil wire with insulation in mm
    :return: inductance, total number of turns (axial * radial), wall thickness, DC resistance, required wire length
    """
    D = float(inner_diameter) / 10
    lk = float(length) / 10
    dw = float(wire_without_insulation_thickness) / 10
    k = float(wire_with_insulation_thickness) / 10
    Ltotal = 0
    lw = 0
    r0 = (D + k) / 2
    N = 0
    Nl = math.floor(lk / k)
    g = math.exp(-0.25) * dw / 2
    nLayer = 0
    while nLayer < layers_count:
        N = N + 1
        Nc = (N - 1) % Nl
        nLayer = math.floor((N - 1) / Nl)
        nx = Nc * k
        ny = r0 + k * nLayer
        Lns = mutual_inductance(ny, ny, g, 0)  # self-inductance of turn
        lw = lw + 2 * math.pi * ny
        M = 0
        if N > 1:
            for j in range(N, 1, -1):
                Jc = (j - 2) % Nl
                jx = Jc * k
                jLayer = math.floor((j - 2) / Nl)
                jy = r0 + k * jLayer
                M = M + 2 * mutual_inductance(
                    ny, jy, nx - jx, g
                )  # Mutual inductance between current and each other turn
        Ltotal = Ltotal + Lns + M

    lw0 = lw / 100
    lw0 = round(lw0 * 1000) / 1000
    R = (0.0175 * lw * 1e-4 * 4) / (math.pi * dw * dw)
    R0 = round(R * 100) / 100
    b = round(math.ceil(nLayer + 1) * k * 1000) / 100

    return ResultingCoilParameters(Ltotal, N, b, R0, lw0)
