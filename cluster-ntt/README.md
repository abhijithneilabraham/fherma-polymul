# Cluster-resident polynomial transforms

Eight thread blocks share one prime's two input planes. Each block holds 32 input columns. It runs the odd-root forward transforms in registers and publishes two tiles to distributed shared memory. One warp then reads a full row across the eight blocks, performs both middle transforms and multiplies the operands. After the inverse middle transform it writes that row back to its owners. Each owner finishes the inverse outer transform and writes the weighted output.

At length 65,536 each block uses 64 KiB of dynamic shared memory and 1,024 threads. Two cluster barriers separate the three stages. The XOR layout keeps each remote row aligned and distributes both access directions across the shared-memory banks. At length 32,768 each block uses 32 KiB.

Compared with the odd-root three-kernel schedule, this removes 21.375 GiB of logical global writes and reads per batch. The new schedule issues the corresponding exchanges through shared memory, reduces the number of independent thread blocks, and may lose on latency or occupancy. Native correctness, cluster scheduling and speed need official measurement.
