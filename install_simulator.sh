#!/bin/sh
set -e

# Cap parallel build jobs by AVAILABLE RAM, never by core count alone: (avail_mb - 6144) / 2560.
# Heavy C++ TUs peak ~2.5GB each; reserve 6GB for whatever else the box is running and hard-cap at 12.
# Uncapped -j$(nproc) on a many-core box is a 48-64GB spike that can take the whole host down.
_mem_mb=$(free -m | awk '/^Mem:/{print $7}')
BUILD_JOBS=$(( (_mem_mb - 6144) / 2560 ))
[ "$BUILD_JOBS" -lt 1 ] && BUILD_JOBS=1
_nproc=$(nproc)
[ "$BUILD_JOBS" -gt "$_nproc" ] && BUILD_JOBS=$_nproc
[ "$BUILD_JOBS" -gt 12 ] && BUILD_JOBS=12
echo "Build parallelism: -j${BUILD_JOBS}  (${_mem_mb}MB available, ${_nproc} cores)"

echo $PATH
echo 'apt update'
apt-get update

cd simulate/trading
if [ ! -d "vcpkg" ]; then
	git clone https://github.com/microsoft/vcpkg.git
fi
cd vcpkg && git reset --hard e140b1fde236eb682b0d47f905e65008a191800f && cd ..
apt-get install -y curl zip unzip tar make pkg-config autoconf autoconf-archive libcurl4-openssl-dev
./vcpkg/bootstrap-vcpkg.sh -disableMetrics

python -m pip install -e .

. /etc/lsb-release
echo "Ubuntu Version $DISTRIB_RELEASE"
if ! g++ -dumpversion | grep -q "14"; then
	echo "g++ is not at version 14!  Checking for g++-14.."
	if ! command -v g++-14
	then
		echo "g++-14 is not available.  Installing.."
		if [ "$DISTRIB_RELEASE" = "22.04" ]; then
				# Clean any leftover from a previously interrupted source build.
				rm -rf gcc-14.1.0 gcc-14.1.0.tar.gz
				# Fast path: ubuntu-toolchain-r PPA ships g++-14 as a binary package (~1 min).
				# Fall back to from-source build (2+ hr) if the PPA is unavailable.
				apt-get install -y software-properties-common
				_gxx14_installed=0
				if add-apt-repository -y ppa:ubuntu-toolchain-r/test \
				   && apt-get update -qq \
				   && apt-get install -y g++-14 libstdc++-14-dev \
				   && command -v g++-14 >/dev/null 2>&1; then
					_gxx14_installed=1
					echo "g++-14 installed from ubuntu-toolchain-r PPA."
				else
					echo "PPA install failed — falling back to gcc-14.1.0 source build."
				fi
				if [ "$_gxx14_installed" = "0" ]; then
					apt-get install libmpfr-dev libgmp3-dev libmpc-dev -y
					wget http://ftp.gnu.org/gnu/gcc/gcc-14.1.0/gcc-14.1.0.tar.gz
					tar -xf gcc-14.1.0.tar.gz
					cd gcc-14.1.0
					./configure -v --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu --prefix=/usr/local/gcc-14.1.0 --enable-checking=release --enable-languages=c,c++ --disable-multilib --program-suffix=-14.1.0
					make -j"$BUILD_JOBS"
					make install
					cd ..
					rm -r gcc-14.1.0
					rm gcc-14.1.0.tar.gz
					update-alternatives --install /usr/bin/g++-14 g++-14 /usr/local/gcc-14.1.0/bin/g++-14.1.0 14
					export LD_LIBRARY_PATH="/usr/local/gcc-14.1.0/lib/../lib64:$LD_LIBRARY_PATH"
					echo 'export LD_LIBRARY_PATH="/usr/local/gcc-14.1.0/lib/../lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
				fi
		else
			apt-get -y install g++-14
		fi
	else
		echo "g++-14 is already available."
	fi
fi

if ! cmake --version | grep -q "3.29.7"; then
	echo "Installing cmake 3.29.7..."
	apt-get purge -y cmake
	# Clean any leftover from a previously interrupted install (prebuilt or source).
	rm -rf cmake-3.29.7 cmake-3.29.7.tar.gz \
	       cmake-3.29.7-linux-x86_64 cmake-3.29.7-linux-x86_64.tar.gz \
	       cmake-3.29.7-linux-aarch64 cmake-3.29.7-linux-aarch64.tar.gz
	# Fast path: Kitware ships prebuilt binaries (~30 s).  Fall back to from-source
	# build (5–15 min serial, ~2–3 min with -j) if the arch isn't supported or the
	# download fails.
	_cmake_arch=""
	case "$(uname -m)" in
		x86_64)  _cmake_arch=linux-x86_64 ;;
		aarch64) _cmake_arch=linux-aarch64 ;;
	esac
	_cmake_installed=0
	if [ -n "$_cmake_arch" ]; then
		echo "Trying prebuilt cmake binary (cmake-3.29.7-${_cmake_arch})..."
		if wget -q "https://github.com/Kitware/CMake/releases/download/v3.29.7/cmake-3.29.7-${_cmake_arch}.tar.gz" \
		   && tar zxf "cmake-3.29.7-${_cmake_arch}.tar.gz" \
		   && cp -r "cmake-3.29.7-${_cmake_arch}/." /usr/local/ \
		   && /usr/local/bin/cmake --version | grep -q "3.29.7"; then
			_cmake_installed=1
			echo "Prebuilt cmake 3.29.7 installed."
		else
			echo "Prebuilt cmake install failed — falling back to source build."
		fi
		rm -rf "cmake-3.29.7-${_cmake_arch}" "cmake-3.29.7-${_cmake_arch}.tar.gz"
	fi
	if [ "$_cmake_installed" = "0" ]; then
		wget https://github.com/Kitware/CMake/releases/download/v3.29.7/cmake-3.29.7.tar.gz
		tar zxvf cmake-3.29.7.tar.gz
		cd cmake-3.29.7
		./bootstrap --parallel="$(nproc)"
		make -j"$BUILD_JOBS"
		make install
		cd ..
		rm cmake-3.29.7.tar.gz
		rm -r cmake-3.29.7
	fi
else
	echo "cmake 3.29.7 is already installed."
fi

rm -r build || true
mkdir build
cd build
if ! g++ -dumpversion | grep -q "14"; then
	cmake -DCMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=g++-14 ..
else
	cmake -DCMAKE_BUILD_TYPE=Release ..
fi
cmake --build . -j "$BUILD_JOBS"