"""
    _resolved_ldlt_block(MultiFloat{T,N}, config)

Measured dense LDLT panel widths for one- through four-limb MultiFloats. On the
four-thread x86-64 scaling gate, 16/12/8 columns were the stable winners for
x2/x3/x4 at both n=512 and n=1024. An explicit `ldlt_block` always overrides
these defaults, and `ldlt_plan` still controls the crossover independently.
"""
@inline function _resolved_ldlt_block(
    ::Type{MultiFloat{T,N}},
    config::KernelConfig,
) where {T,N}
    config.ldlt_block > 0 && return config.ldlt_block
    return N <= 2 ? 16 : N == 3 ? 12 : 8
end
