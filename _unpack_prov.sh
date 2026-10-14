#!/usr/bin/env sh
# 用法：把 **两个** 文件拖进 JupyterLab 的 /root/autodl-tmp/ 目录 ——
#         · prov_upload.tar.gz
#         · _unpack_prov.sh（本文件自己也要在盒子上）
#       然后：  sh /root/autodl-tmp/_unpack_prov.sh
# 清单在 tar 内部，**不需要**第三个文件。
set -e
rm -rf /tmp/pp.new && mkdir -p /tmp/pp.new
tar xzf /root/autodl-tmp/prov_upload.tar.gz -C /tmp/pp.new
( cd /tmp/pp.new && sha256sum -c paper_experiments.sha256 )
/root/miniconda3/envs/mamba2/bin/python -m py_compile /tmp/pp.new/paper_experiments.py
for f in /tmp/pp.new/*.sh; do bash -n "$f" || { echo "★ 语法错 $f"; exit 1; }; done
cd /root/autodl-tmp
if [ -e paper_experiments.py ]; then cp -n paper_experiments.py "paper_experiments.py.bak-$(date +%Y%m%d-%H%M%S)"; fi
mv /tmp/pp.new/paper_experiments.py paper_experiments.py
mv /tmp/pp.new/*.sh /root/autodl-tmp/
chmod +x /root/autodl-tmp/*.sh
echo "★ 就位：paper_experiments.py + box_smoke.sh box_reprov.sh box_fetch.sh（旧版已备份为 .bak-日期-时分秒）"
echo "  盒子上的 sha256（应与本地相同）："
sha256sum /root/autodl-tmp/paper_experiments.py
