import os
import pydicom
from glob import glob

import datetime

class DicomTagUtils:
    def __init__(self, dicom_file, is_series=False, postfix='', min_dicom_nums=10):
        if is_series:
            dcm_files = glob(os.path.join(dicom_file, '*{}'.format(postfix)))
            dcm_files.sort()
            cur_dcm_file = dcm_files[0]
        else:
            cur_dcm_file = dicom_file

        self.metadata = pydicom.dcmread(cur_dcm_file)

    @staticmethod
    def load_metadata(dicom_file, is_series=False, postfix='', min_dicom_nums=10):
        if is_series:
            dcm_files = glob(os.path.join(dicom_file, '*{}'.format(postfix)))
            dcm_files.sort()
            cur_dcm_file = dcm_files[0]
        else:
            cur_dcm_file = dicom_file

        metadata = pydicom.dcmread(cur_dcm_file)
        return metadata

    @staticmethod
    def get_basic_info(info):
        basic_info = {}
        acq_time = DicomTagUtils.get_dicom_acq_datetime(info)
        series_uid = info.SeriesInstanceUID if 'SeriesInstanceUID' in info else ''
        desc = info.SeriesDescription if 'SeriesDescription' in info else ''
        modality = info.Modality if 'Modality' in info else ''
        age = info.PatientAge if 'PatientAge' in info else ''
        sex = info.PatientSex if 'PatientSex' in info else ''
        pos = info.PatientPosition if 'PatientPosition' in info else ''
        pid = info.PatientID if 'PatientID' in info else ''
        study_desc = info.StudyDescription if 'StudyDescription' in info else ''
        study_uid = info.StudyInstanceUID if 'StudyInstanceUID' in info else ''
        
        basic_info['acq_time'] = acq_time
        basic_info['series_uid'] = series_uid
        basic_info['desc'] = desc
        basic_info['modality'] = modality
        basic_info['age'] = age
        basic_info['sex'] = sex
        basic_info['pos'] = pos
        basic_info['pid'] = pid
        basic_info['study_desc'] = study_desc
        basic_info['study_uid'] = study_uid

        return basic_info
    

    @staticmethod
    def get_dicom_acq_datetime(metadata):
        if 'AcquisitionDate' in metadata and 'AcquisitionTime' in metadata:
            acq_time = metadata.AcquisitionDate + metadata.AcquisitionTime
        elif 'AcquisitionDateTime' in metadata:
            acq_time = metadata.AcquisitionDateTime
        else:
            acq_time = ''

        dt_acq_time = datetime.datetime.strptime(acq_time.split('.')[0], '%Y%m%d%H%M%S')

        return dt_acq_time

    



def test_DicomTagUtils():
    # dicom_series = '/fileser/zhangwd/data/brain/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞\(188例）/六院-DWI闭塞病例\(105\)/1014186/CTA'
    dicom_series = '/fileser/zhangwd/data/brain/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-DWI闭塞病例(105)/1014186/CTA'

    metadata = DicomTagUtils.load_metadata(dicom_series, is_series=True)
    DicomTagUtils.get_basic_info(metadata)

    print('finished test_DicomTagUtils!')



if __name__ == '__main__':
    test_DicomTagUtils()

