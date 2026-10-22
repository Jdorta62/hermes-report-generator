# Manual de operación de Faro

Faro es el sistema de monitorización de la red de sensores de Nébula Logística. Recoge lecturas de temperatura y humedad de los almacenes cada 30 segundos.

## Instalación

Faro se instala con el paquete faro-server en la versión 3.2 o superior. El servidor necesita al menos 8 GB de memoria y un disco de 200 GB dedicado a las lecturas.

## Copias de seguridad

Las copias de seguridad protegen la base de datos de lecturas y la configuración de los sensores. Se guardan en el bucket interno faro-respaldos.

Faro hace una copia de seguridad incremental cada 6 horas y una copia completa cada domingo a las 03:00. Las copias incrementales tardan unos 4 minutos.

Durante la copia completa el panel web pasa a modo de solo lectura, pero la recogida de lecturas no se interrumpe.

Las copias incrementales se conservan 14 días y las completas 90 días. Pasado ese plazo se borran de forma automática.

## Restauración

Para restaurar una copia se usa el comando faroctl restaurar --punto <id>. El identificador de cada punto de restauración aparece con faroctl puntos.

## Alertas

Faro envía una alerta al canal #faro-guardia cuando un sensor deja de enviar lecturas durante más de 5 minutos.
