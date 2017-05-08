The ChatServer is written using the POCO C++ library.
The sample fetches the POCO dependencies by referencing the Fix8 POCO dependency NuGet package: https://www.nuget.org/packages/fix8.dependencies.poco/

Here is the folder structure of the POCO dependencies (required by the application):
poco\
    poco110
           bin\PocoFoundation.dll
           bin\PocoFoundationd.dll
           bin\PocoNet.dll
           bin\PocoNetd.dll
           bin64\PocoFoundation64.dll
           bin64\PocoFoundationd64.dll
           bin64\PocoNet64.dll
           bin64\PocoNetd64.dll
           include\Foundation\include\Poco\*.h (POCO Foundation headers)
           include\Net\include\Poco\Net\*.h (POCO Net headers)
           lib\PocoFoundation.lib
           lib\PocoFoundationd.lib
           lib\PocoNet.lib
           lib\PocoNetd.lib
           lib64\PocoFoundation64.lib
           lib64\PocoFoundationd64.lib
           lib64\PocoNet64.lib
           lib64\PocoNetd64.lib

Alternatively, you can also change the include directiories of the ChatServer project to include POCO Foundation and POCO Net headers and update linker properties to link to PocoFoundation.dll and PocoNet.dll.