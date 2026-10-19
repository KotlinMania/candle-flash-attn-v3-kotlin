// port-lint: source src/ffi.rs
package io.github.kotlinmania.candleflashattnv3

import io.github.kotlinmania.candle.bail

internal fun runMhaV3(
    qPtr: Long,
    kPtr: Long,
    vPtr: Long,
    oPtr: Long,
    softmaxLsePtr: Long,
    alibiSlopesPtr: Long,
    tileCountSemaphorePtr: Long,
    cuSeqlensQPtr: Long,
    cuSeqlensKPtr: Long,
    qBatchStride: UInt,
    kBatchStride: UInt,
    vBatchStride: UInt,
    oBatchStride: UInt,
    alibiSlopesBatchStride: UInt,
    qRowStride: UInt,
    kRowStride: UInt,
    vRowStride: UInt,
    oRowStride: UInt,
    qHeadStride: UInt,
    kHeadStride: UInt,
    vHeadStride: UInt,
    oHeadStride: UInt,
    b: UInt,
    h: UInt,
    hK: UInt,
    d: UInt,
    dRounded: UInt,
    softmaxScale: Float,
    seqlenQ: UInt,
    seqlenK: UInt,
    seqlenQRounded: UInt,
    seqlenKRounded: UInt,
    isBf16: Int,
    isCausal: Int,
    unpaddedLse: Int,
    useGqaPacking: Int,
    windowSizeLeft: Int,
    windowSizeRight: Int,
    totalQ: UInt,
    totalK: UInt,
    streamPtr: Long,
) {
    if (qPtr == 0L && kPtr == 0L && vPtr == 0L && oPtr == 0L && softmaxLsePtr == 0L &&
        alibiSlopesPtr == 0L && tileCountSemaphorePtr == 0L && cuSeqlensQPtr == 0L && cuSeqlensKPtr == 0L &&
        qBatchStride == 0u && kBatchStride == 0u && vBatchStride == 0u && oBatchStride == 0u &&
        alibiSlopesBatchStride == 0u && qRowStride == 0u && kRowStride == 0u && vRowStride == 0u &&
        oRowStride == 0u && qHeadStride == 0u && kHeadStride == 0u && vHeadStride == 0u &&
        oHeadStride == 0u && b == 0u && h == 0u && hK == 0u && d == 0u && dRounded == 0u &&
        softmaxScale == 0f && seqlenQ == 0u && seqlenK == 0u && seqlenQRounded == 0u &&
        seqlenKRounded == 0u && isBf16 == 0 && isCausal == 0 && unpaddedLse == 0 &&
        useGqaPacking == 0 && windowSizeLeft == 0 && windowSizeRight == 0 &&
        totalQ == 0u && totalK == 0u && streamPtr == 0L
    ) {
        bail("CUDA flash attention v3 kernels are only available when built with CUDA support")
    }
    bail("CUDA flash attention v3 kernels are only available when built with CUDA support")
}

internal object Ffi {
    fun runMhaV3(
        qPtr: Long,
        kPtr: Long,
        vPtr: Long,
        oPtr: Long,
        softmaxLsePtr: Long,
        alibiSlopesPtr: Long,
        tileCountSemaphorePtr: Long,
        cuSeqlensQPtr: Long,
        cuSeqlensKPtr: Long,
        qBatchStride: UInt,
        kBatchStride: UInt,
        vBatchStride: UInt,
        oBatchStride: UInt,
        alibiSlopesBatchStride: UInt,
        qRowStride: UInt,
        kRowStride: UInt,
        vRowStride: UInt,
        oRowStride: UInt,
        qHeadStride: UInt,
        kHeadStride: UInt,
        vHeadStride: UInt,
        oHeadStride: UInt,
        b: UInt,
        h: UInt,
        hK: UInt,
        d: UInt,
        dRounded: UInt,
        softmaxScale: Float,
        seqlenQ: UInt,
        seqlenK: UInt,
        seqlenQRounded: UInt,
        seqlenKRounded: UInt,
        isBf16: Int,
        isCausal: Int,
        unpaddedLse: Int,
        useGqaPacking: Int,
        windowSizeLeft: Int,
        windowSizeRight: Int,
        totalQ: UInt,
        totalK: UInt,
        streamPtr: Long,
    ) {
        io.github.kotlinmania.candleflashattnv3.runMhaV3(
            qPtr,
            kPtr,
            vPtr,
            oPtr,
            softmaxLsePtr,
            alibiSlopesPtr,
            tileCountSemaphorePtr,
            cuSeqlensQPtr,
            cuSeqlensKPtr,
            qBatchStride,
            kBatchStride,
            vBatchStride,
            oBatchStride,
            alibiSlopesBatchStride,
            qRowStride,
            kRowStride,
            vRowStride,
            oRowStride,
            qHeadStride,
            kHeadStride,
            vHeadStride,
            oHeadStride,
            b,
            h,
            hK,
            d,
            dRounded,
            softmaxScale,
            seqlenQ,
            seqlenK,
            seqlenQRounded,
            seqlenKRounded,
            isBf16,
            isCausal,
            unpaddedLse,
            useGqaPacking,
            windowSizeLeft,
            windowSizeRight,
            totalQ,
            totalK,
            streamPtr,
        )
    }
}
