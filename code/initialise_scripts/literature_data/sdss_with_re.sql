-- This query finds the mass of elliptical galaxies > 9 dex in SDSS
SELECT TOP 1000
    p.specobjid, p.logMass, a.deVRad_u
FROM stellarMassFSPSGranEarlyDust AS p
   LEFT JOIN zooSpec AS s ON s.specobjid = p.specobjid
   LEFT JOIN PhotoObjAll AS a ON a.objID = p.specobjid
WHERE 
    s.elliptical=1
    AND p.logMass > 9