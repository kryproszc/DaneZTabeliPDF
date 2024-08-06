sudo ip addr add 192.168.137.2/24 dev eth0
sudo ip link set eth0 up
sudo ip route add default via 192.168.137.1


ping 192.168.137.1

  ping 192.168.137.2

  
$SwitchInterface = Get-NetAdapter | Where-Object {$_.Name -like "vEthernet (WSLSwitch)"}

$PhysicalInterfaceName = "Ethernet0 2"
 
Set-NetIPInterface -InterfaceAlias $SwitchInterface.Name -Forwarding Enabled
Set-NetIPInterface -InterfaceAlias $PhysicalInterfaceName -Forwarding Enabled

 
New-NetNat -Name WSLNat -InternalIPInterfaceAddressPrefix 192.168.137.0/24 -ExternalIPInterface $PhysicalInterfaceName
