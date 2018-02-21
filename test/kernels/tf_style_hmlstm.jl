#########################################
# TF-style HMLSTM gradient calculations #
#########################################

function tf_hmlstm_update_c_gradients(z, zb, c, f, i, g)
    p0, p1, p2, p3, p4, p5 = , z, zb, , g, # TODO unpack variables
    tanhp4 = tanh.(p4)
    fusion2 = tf_fusion_2_or_5(p5)
    fusion5 = tf_fusion_2_or_5(p3)
    fusion1 = tf_fusion1(fusion2, tanhp4, p1, p2)
    fusion = tf_fusion(fusion2, tanhp4, p1, p2)
    fusion3 = tf_fusion3(fusion5, p0, p1, p2)
    fusion4 = tf_fusion4(fusion5, p1, p2)
    ∇c, ∇f, ∇i, ∇g = # TODO unpack variables
    return ∇c, ∇f, ∇i, ∇g
end

function tf_fusion(fusion2, tanhp4, p1, p2)
    p5 = p1
    p4 = p2
    p3 = 0.0f0
    p2 = 1.0f0
    p1 = fusion2
    p0 = tanhp4
    return broadcast(p0, p1, p2, p3, p4, p5) do x0, x1, x2, x3, x4, x5
        # TODO
    end
end

function tf_fusion1(fusion2, tanhp4, p1, p2)
    p5 = p1
    p4 = p2
    p3 = 0.0f0
    p2 = 1.0f0
    p1 = tanhp4
    p0 = fusion2
    return broadcast(p0, p1, p2, p3, p4, p5) do x0, x1, x2, x3, x4, x5
        # TODO
    end
end

function tf_fusion3(fusion5, p0, p1, p2)
    p5 = p1
    p4 = p2
    p3 = 0.0f0
    p2 = 1.0f0
    p1 = p0
    p0 = fusion5
    return broadcast(p0, p1, p2, p3, p4, p5) do x0, x1, x2, x3, x4, x5
        # TODO
    end
end

function tf_fusion4(fusion5, p1, p2)
    p4 = p1
    p3 = p2
    p2 = 0.0f0
    p1 = 1.0f0
    p0 = fusion5
    return broadcast(p0, p1, p2, p3, p4) do x0, x1, x2, x3, x4
        # TODO
    end
end

# fusion.2 and fusion.5 are exactly the same,
# so we just use this method for both kernels
function tf_fusion_2_or_5(p_3_or_5)
    p1 = 0.5f0
    p0 = p_3_or_5
    return broadcast(p1, p0) do x0, x1
        return x0 + (x0 * tanh(x0 * x1))
    end
end
