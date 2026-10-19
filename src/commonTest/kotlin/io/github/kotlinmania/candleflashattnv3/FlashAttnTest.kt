// port-lint: tests tests/flash_attn_tests.rs
package io.github.kotlinmania.candleflashattnv3

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.i
import io.github.kotlinmania.candle.toVec0
import io.github.kotlinmania.candle.toVec3Round
import io.github.kotlinmania.candlenn.softmax
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

private fun toVec3Round(t: Tensor, digits: Int): List<List<List<Float>>> =
    io.github.kotlinmania.candle.toVec3Round(t, digits)

private fun faAcausal(q: Tensor, k: Tensor, v: Tensor, softmaxScale: Float): Tensor {
    val inDtype = q.dtype()
    val qF32 = q.toDtype(DType.F32)
    val kF32 = k.toDtype(DType.F32)
    val vF32 = v.toDtype(DType.F32)
    val att = qF32.matmul(kF32.t()).mul(softmaxScale.toDouble())
    val attSoftmax = softmax(att, -1)
    val output = attSoftmax.matmul(vF32.contiguous()).toDtype(inDtype)
    return output
}

class FlashAttnTest {
    @Test
    fun flashAttnAcausal() {
        val device = try {
            Device.newCuda(0)
        } catch (_: CandleException) {
            return
        }
        val qRaw = try {
            Tensor.arange(0u, 3u * 2u * 64u, device)
        } catch (_: CandleException) {
            return
        }
        val q = qRaw.toDtype(DType.F16).reshape(Shape(1, 3, 2, 64))
        val k = q / 400.0
        val v = q / 500.0
        val qScaled = q / 300.0

        val ys1 = faAcausal(qScaled, k, v, 0.5f)
        val ys1F32 = ys1.i(0).toDtype(DType.F32)
        val ys2 = run {
            val qT = qScaled.transpose(1, 2)
            val kT = k.transpose(1, 2)
            val vT = v.transpose(1, 2)
            flashAttn(qT, kT, vT, 0.5f, causal = false, useGqaPacking = false).transpose(1, 2)
        }
        val ys2F32 = ys2.i(0).toDtype(DType.F32)
        val diff = (ys1F32 - ys2F32).abs().flattenAll().max(0)

        assertEquals(listOf(3, 2, 64), ys2F32.dims())
        val expected = listOf(
            listOf(
                listOf(
                    0.0808f, 0.0828f, 0.0848f, 0.0869f, 0.0889f, 0.0908f, 0.0928f, 0.0948f, 0.0969f, 0.0989f,
                    0.1008f, 0.1028f, 0.1049f, 0.1069f, 0.1088f, 0.1108f, 0.1129f, 0.1149f, 0.1168f, 0.1188f,
                    0.1208f, 0.1229f, 0.1249f, 0.1268f, 0.1288f, 0.1309f, 0.1328f, 0.1349f, 0.1368f, 0.1388f,
                    0.1409f, 0.1428f, 0.1449f, 0.1469f, 0.1488f, 0.1509f, 0.1528f, 0.1548f, 0.1569f, 0.1588f,
                    0.1609f, 0.1628f, 0.1648f, 0.1669f, 0.1688f, 0.1709f, 0.1729f, 0.1748f, 0.1769f, 0.1788f,
                    0.1809f, 0.1829f, 0.1848f, 0.1869f, 0.1888f, 0.1908f, 0.1929f, 0.1948f, 0.1969f, 0.1989f,
                    0.2008f, 0.2029f, 0.205f, 0.2069f,
                ),
                listOf(
                    0.1071f, 0.1091f, 0.1111f, 0.113f, 0.1151f, 0.1171f, 0.1191f, 0.1211f, 0.123f, 0.1251f,
                    0.1271f, 0.129f, 0.1311f, 0.1331f, 0.135f, 0.1371f, 0.139f, 0.1411f, 0.1431f, 0.145f,
                    0.1471f, 0.149f, 0.1511f, 0.1531f, 0.155f, 0.1571f, 0.1591f, 0.1611f, 0.1631f, 0.165f,
                    0.1671f, 0.1691f, 0.1711f, 0.1731f, 0.175f, 0.1771f, 0.1791f, 0.181f, 0.1831f, 0.1851f,
                    0.1871f, 0.1891f, 0.191f, 0.1931f, 0.1951f, 0.1971f, 0.1991f, 0.201f, 0.2031f, 0.2051f,
                    0.2072f, 0.2091f, 0.2111f, 0.2131f, 0.2151f, 0.217f, 0.2191f, 0.2211f, 0.2231f, 0.2251f,
                    0.2271f, 0.229f, 0.2312f, 0.2332f,
                ),
            ),
            listOf(
                listOf(
                    0.3765f, 0.3784f, 0.3804f, 0.3823f, 0.3843f, 0.3862f, 0.3884f, 0.3904f, 0.3923f, 0.3943f,
                    0.3962f, 0.3984f, 0.4004f, 0.4023f, 0.4043f, 0.4063f, 0.4084f, 0.4104f, 0.4124f, 0.4143f,
                    0.4163f, 0.4185f, 0.4204f, 0.4224f, 0.4243f, 0.4263f, 0.4285f, 0.4304f, 0.4324f, 0.4343f,
                    0.4363f, 0.4385f, 0.4404f, 0.4424f, 0.4443f, 0.4463f, 0.4485f, 0.4504f, 0.4524f, 0.4543f,
                    0.4563f, 0.4585f, 0.4604f, 0.4624f, 0.4644f, 0.4663f, 0.4683f, 0.4705f, 0.4724f, 0.4744f,
                    0.4763f, 0.4783f, 0.4805f, 0.4824f, 0.4844f, 0.4863f, 0.4883f, 0.4905f, 0.4922f, 0.4946f,
                    0.4966f, 0.4985f, 0.5005f, 0.5024f,
                ),
                listOf(
                    0.3816f, 0.3835f, 0.3855f, 0.3875f, 0.3894f, 0.3914f, 0.3936f, 0.3955f, 0.3975f, 0.3994f,
                    0.4014f, 0.4036f, 0.4055f, 0.4075f, 0.4094f, 0.4114f, 0.4136f, 0.4155f, 0.4175f, 0.4194f,
                    0.4214f, 0.4236f, 0.4255f, 0.4275f, 0.4294f, 0.4314f, 0.4336f, 0.4355f, 0.4375f, 0.4395f,
                    0.4414f, 0.4436f, 0.4456f, 0.4475f, 0.4495f, 0.4514f, 0.4536f, 0.4556f, 0.4575f, 0.4595f,
                    0.4614f, 0.4636f, 0.4656f, 0.4675f, 0.4695f, 0.4714f, 0.4734f, 0.4756f, 0.4775f, 0.4795f,
                    0.4814f, 0.4834f, 0.4856f, 0.4875f, 0.4895f, 0.4915f, 0.4934f, 0.4956f, 0.4973f, 0.4998f,
                    0.5015f, 0.5034f, 0.5054f, 0.5073f,
                ),
            ),
            listOf(
                listOf(
                    0.6392f, 0.6411f, 0.6431f, 0.6455f, 0.6475f, 0.6494f, 0.6514f, 0.6533f, 0.6553f, 0.6572f,
                    0.6592f, 0.6611f, 0.6631f, 0.6655f, 0.6675f, 0.6694f, 0.6714f, 0.6733f, 0.6753f, 0.6772f,
                    0.6792f, 0.6812f, 0.6831f, 0.6851f, 0.6875f, 0.6895f, 0.6914f, 0.6934f, 0.6953f, 0.6973f,
                    0.6992f, 0.7012f, 0.7031f, 0.7051f, 0.7075f, 0.7095f, 0.7114f, 0.7134f, 0.7153f, 0.7173f,
                    0.7192f, 0.7212f, 0.7231f, 0.7251f, 0.7275f, 0.7295f, 0.7314f, 0.7334f, 0.7354f, 0.7373f,
                    0.7393f, 0.7412f, 0.7432f, 0.7451f, 0.7476f, 0.7495f, 0.7515f, 0.7534f, 0.7554f, 0.7573f,
                    0.7593f, 0.7612f, 0.7632f, 0.7651f,
                ),
                listOf(
                    0.6396f, 0.6416f, 0.6436f, 0.646f, 0.6479f, 0.6499f, 0.6519f, 0.6538f, 0.6558f, 0.6577f,
                    0.6597f, 0.6616f, 0.6636f, 0.666f, 0.668f, 0.6699f, 0.6719f, 0.6738f, 0.6758f, 0.6777f,
                    0.6797f, 0.6816f, 0.6836f, 0.6855f, 0.688f, 0.6899f, 0.6919f, 0.6938f, 0.6958f, 0.6978f,
                    0.6997f, 0.7017f, 0.7036f, 0.7056f, 0.708f, 0.71f, 0.7119f, 0.7139f, 0.7158f, 0.7178f,
                    0.7197f, 0.7217f, 0.7236f, 0.7256f, 0.728f, 0.73f, 0.7319f, 0.7339f, 0.7358f, 0.7378f,
                    0.7397f, 0.7417f, 0.7437f, 0.7456f, 0.748f, 0.75f, 0.752f, 0.7539f, 0.7559f, 0.7578f,
                    0.7598f, 0.7617f, 0.7637f, 0.7656f,
                ),
            ),
        )
        assertEquals(expected, toVec3Round(ys2F32, 4))
        assertTrue(abs(diff.toVec0<Float>()) < 1e-5f)
    }

    @Test
    fun flashAttnAcausalGqa() {
        val device = try {
            Device.newCuda(0)
        } catch (_: CandleException) {
            return
        }
        val nH = 4
        val nHk = 1

        val qRaw = try {
            Tensor.arange(0u, (nH * 2 * 64).toUInt(), device)
        } catch (_: CandleException) {
            return
        }
        val q = qRaw.toDtype(DType.F16).reshape(Shape(1, nH, 2, 64))
        val gqa = q.narrow(1, 0, nHk)
        assertEquals(listOf(1, nHk, 2, 64), gqa.dims())

        val qScaled = q / 1000.0
        val kGqa = gqa / 400.0
        val vGqa = gqa / 500.0

        val ys2 = run {
            val qT = qScaled.transpose(1, 2)
            val kGqaT = kGqa.transpose(1, 2)
            val vGqaT = vGqa.transpose(1, 2)
            flashAttn(qT, kGqaT, vGqaT, 0.125f, causal = false, useGqaPacking = true).transpose(1, 2)
        }
        val ys2F32 = ys2.i(0).toDtype(DType.F32)
        assertEquals(listOf(nH, 2, 64), ys2F32.dims())

        val expected = listOf(
            listOf(
                listOf(
                    0.0653f, 0.0673f, 0.0693f, 0.0713f, 0.0734f, 0.0753f, 0.0773f, 0.0793f, 0.0813f, 0.0834f,
                    0.0853f, 0.0873f, 0.0894f, 0.0913f, 0.0933f, 0.0953f, 0.0973f, 0.0994f, 0.1013f, 0.1033f,
                    0.1053f, 0.1073f, 0.1094f, 0.1113f, 0.1133f, 0.1154f, 0.1173f, 0.1194f, 0.1213f, 0.1233f,
                    0.1254f, 0.1273f, 0.1294f, 0.1313f, 0.1333f, 0.1354f, 0.1373f, 0.1393f, 0.1414f, 0.1433f,
                    0.1454f, 0.1473f, 0.1493f, 0.1514f, 0.1533f, 0.1554f, 0.1573f, 0.1593f, 0.1614f, 0.1633f,
                    0.1654f, 0.1674f, 0.1693f, 0.1714f, 0.1733f, 0.1753f, 0.1774f, 0.1793f, 0.1814f, 0.1833f,
                    0.1853f, 0.1874f, 0.1895f, 0.1914f,
                ),
                listOf(
                    0.0679f, 0.0699f, 0.072f, 0.0739f, 0.076f, 0.0779f, 0.0799f, 0.082f, 0.0839f, 0.086f,
                    0.088f, 0.0899f, 0.092f, 0.0939f, 0.0959f, 0.098f, 0.0999f, 0.102f, 0.1039f, 0.106f,
                    0.108f, 0.1099f, 0.112f, 0.114f, 0.1159f, 0.118f, 0.1199f, 0.122f, 0.124f, 0.126f,
                    0.1279f, 0.13f, 0.132f, 0.134f, 0.136f, 0.1379f, 0.14f, 0.142f, 0.144f, 0.146f, 0.1479f,
                    0.1499f, 0.152f, 0.1539f, 0.1559f, 0.158f, 0.1599f, 0.162f, 0.1639f, 0.1659f, 0.168f,
                    0.1699f, 0.172f, 0.174f, 0.1759f, 0.178f, 0.1799f, 0.182f, 0.184f, 0.1859f, 0.188f,
                    0.1899f, 0.192f, 0.194f,
                ),
            ),
            listOf(
                listOf(
                    0.0706f, 0.0725f, 0.0746f, 0.0765f, 0.0786f, 0.0806f, 0.0825f, 0.0846f, 0.0865f, 0.0886f,
                    0.0906f, 0.0925f, 0.0946f, 0.0966f, 0.0985f, 0.1006f, 0.1025f, 0.1046f, 0.1066f, 0.1085f,
                    0.1106f, 0.1125f, 0.1146f, 0.1166f, 0.1185f, 0.1206f, 0.1226f, 0.1246f, 0.1266f, 0.1285f,
                    0.1306f, 0.1326f, 0.1346f, 0.1366f, 0.1385f, 0.1406f, 0.1426f, 0.1445f, 0.1466f, 0.1486f,
                    0.1506f, 0.1526f, 0.1545f, 0.1566f, 0.1586f, 0.1606f, 0.1626f, 0.1646f, 0.1666f, 0.1686f,
                    0.1707f, 0.1726f, 0.1746f, 0.1766f, 0.1786f, 0.1805f, 0.1826f, 0.1846f, 0.1866f, 0.1886f,
                    0.1906f, 0.1925f, 0.1947f, 0.1967f,
                ),
                listOf(
                    0.0731f, 0.0751f, 0.0771f, 0.0791f, 0.0812f, 0.0831f, 0.0851f, 0.0872f, 0.0891f, 0.0912f,
                    0.0931f, 0.0951f, 0.0972f, 0.0991f, 0.1011f, 0.1031f, 0.1051f, 0.1072f, 0.1091f, 0.1111f,
                    0.1132f, 0.1151f, 0.1172f, 0.1191f, 0.1212f, 0.1232f, 0.1251f, 0.1272f, 0.1292f, 0.1311f,
                    0.1332f, 0.1351f, 0.1372f, 0.1392f, 0.1411f, 0.1432f, 0.1451f, 0.1471f, 0.1492f, 0.1511f,
                    0.1532f, 0.1552f, 0.1571f, 0.1592f, 0.1611f, 0.1632f, 0.1652f, 0.1671f, 0.1692f, 0.1711f,
                    0.1732f, 0.1752f, 0.1771f, 0.1792f, 0.1812f, 0.1831f, 0.1852f, 0.1871f, 0.1892f, 0.1912f,
                    0.1931f, 0.1951f, 0.1973f, 0.1992f,
                ),
            ),
            listOf(
                listOf(
                    0.0757f, 0.0776f, 0.0797f, 0.0817f, 0.0837f, 0.0857f, 0.0876f, 0.0897f, 0.0917f, 0.0938f,
                    0.0957f, 0.0977f, 0.0997f, 0.1017f, 0.1036f, 0.1057f, 0.1077f, 0.1097f, 0.1117f, 0.1136f,
                    0.1157f, 0.1177f, 0.1198f, 0.1217f, 0.1237f, 0.1257f, 0.1277f, 0.1298f, 0.1317f, 0.1337f,
                    0.1357f, 0.1377f, 0.1398f, 0.1417f, 0.1437f, 0.1458f, 0.1477f, 0.1497f, 0.1517f, 0.1537f,
                    0.1558f, 0.1577f, 0.1597f, 0.1617f, 0.1637f, 0.1658f, 0.1677f, 0.1697f, 0.1718f, 0.1737f,
                    0.1758f, 0.1777f, 0.1797f, 0.1818f, 0.1837f, 0.1857f, 0.1877f, 0.1897f, 0.1918f, 0.1937f,
                    0.1957f, 0.1976f, 0.1998f, 0.2018f,
                ),
                listOf(
                    0.0782f, 0.0802f, 0.0822f, 0.0842f, 0.0862f, 0.0882f, 0.0902f, 0.0922f, 0.0942f, 0.0963f,
                    0.0982f, 0.1002f, 0.1022f, 0.1042f, 0.1062f, 0.1082f, 0.1102f, 0.1122f, 0.1142f, 0.1162f,
                    0.1182f, 0.1202f, 0.1223f, 0.1242f, 0.1262f, 0.1283f, 0.1302f, 0.1322f, 0.1343f, 0.1362f,
                    0.1383f, 0.1403f, 0.1422f, 0.1443f, 0.1462f, 0.1482f, 0.1503f, 0.1522f, 0.1543f, 0.1563f,
                    0.1582f, 0.1603f, 0.1622f, 0.1643f, 0.1663f, 0.1682f, 0.1703f, 0.1722f, 0.1743f, 0.1763f,
                    0.1782f, 0.1803f, 0.1823f, 0.1843f, 0.1863f, 0.1882f, 0.1903f, 0.1923f, 0.1943f, 0.1963f,
                    0.1982f, 0.2002f, 0.2023f, 0.2043f,
                ),
            ),
            listOf(
                listOf(
                    0.0807f, 0.0826f, 0.0847f, 0.0867f, 0.0887f, 0.0907f, 0.0927f, 0.0947f, 0.0967f, 0.0987f,
                    0.1007f, 0.1027f, 0.1047f, 0.1067f, 0.1086f, 0.1107f, 0.1127f, 0.1147f, 0.1167f, 0.1187f,
                    0.1207f, 0.1227f, 0.1247f, 0.1267f, 0.1287f, 0.1307f, 0.1327f, 0.1348f, 0.1367f, 0.1387f,
                    0.1407f, 0.1427f, 0.1448f, 0.1467f, 0.1487f, 0.1508f, 0.1527f, 0.1547f, 0.1567f, 0.1587f,
                    0.1608f, 0.1627f, 0.1647f, 0.1667f, 0.1687f, 0.1708f, 0.1727f, 0.1747f, 0.1768f, 0.1787f,
                    0.1808f, 0.1827f, 0.1847f, 0.1868f, 0.1887f, 0.1907f, 0.1927f, 0.1947f, 0.1968f, 0.1987f,
                    0.2007f, 0.2026f, 0.2048f, 0.2068f,
                ),
                listOf(
                    0.0831f, 0.0851f, 0.0871f, 0.0891f, 0.0911f, 0.0931f, 0.0951f, 0.0971f, 0.0991f, 0.1011f,
                    0.1031f, 0.1051f, 0.1071f, 0.1091f, 0.1111f, 0.1131f, 0.1151f, 0.1171f, 0.1191f, 0.1211f,
                    0.1231f, 0.1251f, 0.1271f, 0.1292f, 0.1311f, 0.1332f, 0.1351f, 0.1371f, 0.1392f, 0.1411f,
                    0.1432f, 0.1451f, 0.1471f, 0.1492f, 0.1511f, 0.1531f, 0.1552f, 0.1571f, 0.1592f, 0.1611f,
                    0.1631f, 0.1652f, 0.1671f, 0.1692f, 0.1711f, 0.1731f, 0.1752f, 0.1771f, 0.1792f, 0.1812f,
                    0.1831f, 0.1852f, 0.1871f, 0.1891f, 0.1912f, 0.1931f, 0.1952f, 0.1971f, 0.1991f, 0.2012f,
                    0.2031f, 0.2051f, 0.2072f, 0.2092f,
                ),
            ),
        )
        assertEquals(expected, toVec3Round(ys2F32, 4))
    }

    @Test
    fun flashAttnVarlen() {
        val device = try {
            Device.newCuda(0)
        } catch (_: CandleException) {
            return
        }
        val qRaw = try {
            Tensor.arange(0u, 3u * 2u * 64u, device)
        } catch (_: CandleException) {
            return
        }
        val q = qRaw.toDtype(DType.F16).reshape(Shape(3, 2, 64))
        val k = q / 400.0
        val v = q / 500.0
        val qScaled = q / 300.0

        val seqlensQ = Tensor.new(uintArrayOf(0u, 2u), Shape(2), device)

        val ys = run {
            val qT = qScaled.transpose(0, 1)
            val kT = k.transpose(0, 1)
            val vT = v.transpose(0, 1)
            flashAttnVarlen(
                qT,
                kT,
                vT,
                seqlensQ,
                seqlensQ,
                2,
                2,
                0.5f,
                causal = false,
                useGqaPacking = false,
            ).transpose(0, 1)
        }.toDtype(DType.F32)

        assertEquals(listOf(3, 2, 64), ys.dims())
        val expected = listOf(
            listOf(
                listOf(
                    0.0808f, 0.0828f, 0.0848f, 0.0869f, 0.0889f, 0.0908f, 0.0928f, 0.0948f, 0.0969f, 0.0989f,
                    0.1008f, 0.1028f, 0.1049f, 0.1069f, 0.1088f, 0.1108f, 0.1129f, 0.1149f, 0.1168f, 0.1188f,
                    0.1208f, 0.1229f, 0.1249f, 0.1268f, 0.1288f, 0.1309f, 0.1328f, 0.1349f, 0.1368f, 0.1388f,
                    0.1409f, 0.1428f, 0.1449f, 0.1469f, 0.1488f, 0.1509f, 0.1528f, 0.1548f, 0.1569f, 0.1588f,
                    0.1609f, 0.1628f, 0.1648f, 0.1669f, 0.1688f, 0.1709f, 0.1729f, 0.1748f, 0.1769f, 0.1788f,
                    0.1809f, 0.1829f, 0.1848f, 0.1869f, 0.1888f, 0.1908f, 0.1929f, 0.1948f, 0.1969f, 0.1989f,
                    0.2008f, 0.2029f, 0.205f, 0.2069f,
                ),
                listOf(
                    0.1071f, 0.1091f, 0.1111f, 0.113f, 0.1151f, 0.1171f, 0.1191f, 0.1211f, 0.123f, 0.1251f,
                    0.1271f, 0.129f, 0.1311f, 0.1331f, 0.135f, 0.1371f, 0.139f, 0.1411f, 0.1431f, 0.145f,
                    0.1471f, 0.149f, 0.1511f, 0.1531f, 0.155f, 0.1571f, 0.1591f, 0.1611f, 0.1631f, 0.165f,
                    0.1671f, 0.1691f, 0.1711f, 0.1731f, 0.175f, 0.1771f, 0.1791f, 0.181f, 0.1831f, 0.1851f,
                    0.1871f, 0.1891f, 0.191f, 0.1931f, 0.1951f, 0.1971f, 0.1991f, 0.201f, 0.2031f, 0.2051f,
                    0.2072f, 0.2091f, 0.2111f, 0.2131f, 0.2151f, 0.217f, 0.2191f, 0.2211f, 0.2231f, 0.2251f,
                    0.2271f, 0.229f, 0.2312f, 0.2332f,
                ),
            ),
            listOf(
                listOf(
                    0.3765f, 0.3784f, 0.3804f, 0.3823f, 0.3843f, 0.3862f, 0.3884f, 0.3904f, 0.3923f, 0.3943f,
                    0.3962f, 0.3984f, 0.4004f, 0.4023f, 0.4043f, 0.4063f, 0.4084f, 0.4104f, 0.4124f, 0.4143f,
                    0.4163f, 0.4185f, 0.4204f, 0.4224f, 0.4243f, 0.4263f, 0.4285f, 0.4304f, 0.4324f, 0.4343f,
                    0.4363f, 0.4385f, 0.4404f, 0.4424f, 0.4443f, 0.4463f, 0.4485f, 0.4504f, 0.4524f, 0.4543f,
                    0.4563f, 0.4585f, 0.4604f, 0.4624f, 0.4644f, 0.4663f, 0.4683f, 0.4705f, 0.4724f, 0.4744f,
                    0.4763f, 0.4783f, 0.4805f, 0.4824f, 0.4844f, 0.4863f, 0.4883f, 0.4905f, 0.4922f, 0.4946f,
                    0.4966f, 0.4985f, 0.5005f, 0.5024f,
                ),
                listOf(
                    0.3816f, 0.3835f, 0.3855f, 0.3875f, 0.3894f, 0.3914f, 0.3936f, 0.3955f, 0.3975f, 0.3994f,
                    0.4014f, 0.4036f, 0.4055f, 0.4075f, 0.4094f, 0.4114f, 0.4136f, 0.4155f, 0.4175f, 0.4194f,
                    0.4214f, 0.4236f, 0.4255f, 0.4275f, 0.4294f, 0.4314f, 0.4336f, 0.4355f, 0.4375f, 0.4395f,
                    0.4414f, 0.4436f, 0.4456f, 0.4475f, 0.4495f, 0.4514f, 0.4536f, 0.4556f, 0.4575f, 0.4595f,
                    0.4614f, 0.4636f, 0.4656f, 0.4675f, 0.4695f, 0.4714f, 0.4734f, 0.4756f, 0.4775f, 0.4795f,
                    0.4814f, 0.4834f, 0.4856f, 0.4875f, 0.4895f, 0.4915f, 0.4934f, 0.4956f, 0.4973f, 0.4998f,
                    0.5015f, 0.5034f, 0.5054f, 0.5073f,
                ),
            ),
            listOf(
                listOf(
                    0.6392f, 0.6411f, 0.6431f, 0.6455f, 0.6475f, 0.6494f, 0.6514f, 0.6533f, 0.6553f, 0.6572f,
                    0.6592f, 0.6611f, 0.6631f, 0.6655f, 0.6675f, 0.6694f, 0.6714f, 0.6733f, 0.6753f, 0.6772f,
                    0.6792f, 0.6812f, 0.6831f, 0.6851f, 0.6875f, 0.6895f, 0.6914f, 0.6934f, 0.6953f, 0.6973f,
                    0.6992f, 0.7012f, 0.7031f, 0.7051f, 0.7075f, 0.7095f, 0.7114f, 0.7134f, 0.7153f, 0.7173f,
                    0.7192f, 0.7212f, 0.7231f, 0.7251f, 0.7275f, 0.7295f, 0.7314f, 0.7334f, 0.7354f, 0.7373f,
                    0.7393f, 0.7412f, 0.7432f, 0.7451f, 0.7476f, 0.7495f, 0.7515f, 0.7534f, 0.7554f, 0.7573f,
                    0.7593f, 0.7612f, 0.7632f, 0.7651f,
                ),
                listOf(
                    0.6396f, 0.6416f, 0.6436f, 0.646f, 0.6479f, 0.6499f, 0.6519f, 0.6538f, 0.6558f, 0.6577f,
                    0.6597f, 0.6616f, 0.6636f, 0.666f, 0.668f, 0.6699f, 0.6719f, 0.6738f, 0.6758f, 0.6777f,
                    0.6797f, 0.6816f, 0.6836f, 0.6855f, 0.688f, 0.6899f, 0.6919f, 0.6938f, 0.6958f, 0.6978f,
                    0.6997f, 0.7017f, 0.7036f, 0.7056f, 0.708f, 0.71f, 0.7119f, 0.7139f, 0.7158f, 0.7178f,
                    0.7197f, 0.7217f, 0.7236f, 0.7256f, 0.728f, 0.73f, 0.7319f, 0.7339f, 0.7358f, 0.7378f,
                    0.7397f, 0.7417f, 0.7437f, 0.7456f, 0.748f, 0.75f, 0.752f, 0.7539f, 0.7559f, 0.7578f,
                    0.7598f, 0.7617f, 0.7637f, 0.7656f,
                ),
            ),
        )
        assertEquals(expected, toVec3Round(ys, 4))
    }

    private fun testFlashAttnVarlenParam(headDim: Int, seqLen: Int, useGqaPacking: Boolean) {
        val device = try {
            Device.newCuda(0)
        } catch (_: CandleException) {
            return
        }
        val qRaw = try {
            Tensor.arange(0u, (3 * seqLen * headDim).toUInt(), device)
        } catch (_: CandleException) {
            return
        }
        val q = qRaw.toDtype(DType.F16).reshape(Shape(3, seqLen, headDim))
        val denom = (headDim * seqLen).toDouble()
        val k = q / (denom * 4.0)
        val v = q / (denom * 2.0)
        val qScaled = q / (denom * 3.0)

        val seqlensQ = Tensor.new(uintArrayOf(0u, seqLen.toUInt()), Shape(2), device)
        val seqlensK = Tensor.new(uintArrayOf(0u, seqLen.toUInt()), Shape(2), device)

        val ys = run {
            val qT = qScaled.transpose(0, 1)
            val kT = k.transpose(0, 1)
            val vT = v.transpose(0, 1)
            flashAttnVarlen(
                qT,
                kT,
                vT,
                seqlensQ,
                seqlensK,
                maxSeqlenQ = seqLen,
                maxSeqlenK = seqLen,
                softmaxScale = 0.5f,
                causal = false,
                useGqaPacking = useGqaPacking,
            ).transpose(0, 1)
        }.toDtype(DType.F32)

        assertEquals(listOf(3, seqLen, headDim), ys.dims())
        val ys2 = run {
            val qUnsq = qScaled.unsqueeze(0)
            val kUnsq = k.unsqueeze(0)
            val vUnsq = v.unsqueeze(0)
            val y = faAcausal(qUnsq, kUnsq, vUnsq, 0.5f)
            y.i(0).toDtype(DType.F32)
        }
        val diff = (ys - ys2).abs().flattenAll().max(0)
        assertTrue(abs(diff.toVec0<Float>()) < 5e-3f)
    }

    @Test
    fun flashAttnVarlenParam() {
        val headDims = listOf(64, 128, 256, 512)
        val seqLens = listOf(2, 4, 9)
        val useGqaPackings = listOf(false)
        for (headDim in headDims) {
            for (seqLen in seqLens) {
                for (useGqa in useGqaPackings) {
                    testFlashAttnVarlenParam(headDim, seqLen, useGqa)
                }
            }
        }
    }
}
