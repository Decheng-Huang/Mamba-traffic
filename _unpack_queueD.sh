#!/usr/bin/env sh
# 用法：把 queueD.tar.gz 放到 /root/autodl-tmp/，然后
#        sh /root/autodl-tmp/_unpack_queueD.sh
# 清单就在 tar 里（queueD.sha256），**不需要**再传第二个文件。
set -e
rm -rf /tmp/qd.new && mkdir -p /tmp/qd.new
tar xzf /root/autodl-tmp/queueD.tar.gz -C /tmp/qd.new
( cd /tmp/qd.new && sha256sum -c queueD.sha256 )
for f in /tmp/qd.new/queueD/*.sh; do bash -n "$f" || { echo "★ 语法错 $f"; exit 1; }; done
cd /root/autodl-tmp
if [ -e queueD ]; then rm -rf queueD.bak && mv queueD queueD.bak; fi
mv /tmp/qd.new/queueD queueD
echo "★ queueD/ 就位"
ls -l queueD/
