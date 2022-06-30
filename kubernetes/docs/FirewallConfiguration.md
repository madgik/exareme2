## Firewall Configuration

Using firewalld the following rules should apply,

in the **master** node:

```
firewall-cmd --permanent --add-port=6443/tcp       # Kubelet api port
firewall-cmd --permanent --add-port=30000/tcp      # MIPEngine Controller port
```

on all nodes:

```
firewall-cmd --zone=public --permanent --add-rich-rule='rule protocol value="ipip" accept'  # Protocol "4" for "calico"-network-plugin.
```

These rules allow for kubectl to only be run on the **master** node.
