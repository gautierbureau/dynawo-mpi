//
// Copyright (c) 2015-2019, RTE (http://www.rte-france.com)
// See AUTHORS.txt
// All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, you can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
//
// This file is part of Dynawo, an hybrid C++/Modelica open source time domain
// simulation tool for power systems.
//

/**
 * @file  main.cpp
 *
 * @brief main program of dynawo
 *
 */
#include <string>
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>

#include <xml/sax/parser/ParserException.h>

#include "config.h"
#include "gitversion.h"

#include "DYNMacrosMessage.h"
#include "DYNError.h"
#include "DYNIoDico.h"
#include "DYNTrace.h"
#include "DYNFileSystemUtils.h"
#include "DYNExecUtils.h"
#include "DYNInitXml.h"
#include "DYNTimer.h"

#include "DYNSimulation.h"
#include "DYNSimulationContext.h"
#include "JOBXmlImporter.h"
#include "JOBIterators.h"
#include "JOBJobsCollection.h"
#include "JOBJobEntry.h"
#include "JOBOutputsEntry.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using std::string;
using std::exception;
using std::endl;
using std::cerr;
using std::cout;
using std::vector;

using DYN::Error;
using DYN::Trace;
using DYN::IoDico;

namespace po = boost::program_options;

namespace parser = xml::sax::parser;

using boost::shared_ptr;

static void usage(const po::options_description& desc) {
  cout << "Usage: dynawo <jobs-file>" << std::endl << std::endl;
  cout << desc << endl;
}

// If logging is disabled, Trace::info has no effect so we also print on standard output to have basic information
template<class T>
static void print(const T& output, DYN::SeverityLevel level = DYN::INFO) {
  DYN::TraceStream ss;
  switch (level) {
    case DYN::DEBUG:
      ss = Trace::debug();
      break;
    case DYN::INFO:
      ss = Trace::info();
      break;
    case DYN::WARN:
      ss = Trace::warn();
      break;
    case DYN::ERROR:
      ss = Trace::error();
      break;
    default:
      // impossible case by definition of the enum
      return;
  }
  ss << output << Trace::endline;
  if (!Trace::isLoggingEnabled()) {
    std::clog << output << std::endl;
  }
}

void launchSimuLocale(const std::string& jobsFileName);

int main(int argc, char ** argv) {
  string jobsFileName = "";
  int nbThreads = 1;

  // declarations of supported options
  // -----------------------------------

  po::options_description desc;
  desc.add_options()
    ("help,h", "produce help message")
    ("version,v", "print dynawo version")
    ("nbThreads", po::value<int>(&nbThreads), "Set the number of threads that could be used by the simulation");

  po::options_description hidden("Hidden options");
  hidden.add_options() ("jobs-file", po::value<string>(&jobsFileName), "set job file");

  po::positional_options_description positionalOptions;
  positionalOptions.add("jobs-file", 1);

  po::options_description cmdlineOptions;
  cmdlineOptions.add(desc).add(hidden);

  try {
    po::variables_map vm;
    // parse regular options
    po::store(po::command_line_parser(argc, argv).options(cmdlineOptions)
                .positional(positionalOptions).run(),
              vm);
    po::notify(vm);

    if (vm.count("help")) {
      usage(desc);
      return 0;
    }

    if (vm.count("version")) {
      cout << DYNAWO_VERSION_STRING << " (rev:" << DYNAWO_GIT_BRANCH << "-" << DYNAWO_GIT_HASH << ")" << endl;
      return 0;
    }

    // launch simulation
    if (jobsFileName == "") {
      cout << "Error: a jobs file name is required." << endl;
      usage(desc);
      return 1;
    }

    if (!exists(jobsFileName)) {
      cout << " failed to locate jobs file (" << jobsFileName << ")" << endl;
      usage(desc);
      return 1;
    }

#ifdef WITH_OPENMP
    omp_set_num_threads(nbThreads);
#endif

    DYN::InitXerces xerces;
    DYN::InitLibXml2 libxml2;
    DYN::IoDicos& dicos = DYN::IoDicos::instance();
    dicos.addPath(getMandatoryEnvVar("DYNAWO_RESOURCES_DIR"));
    dicos.addDicos(getMandatoryEnvVar("DYNAWO_DICTIONARIES"));
    if (getEnvVar("DYNAWO_USE_XSD_VALIDATION") != "true")
      cout << "[INFO] xsd validation will not be used" << endl;

    Trace::init();
    Trace::resetCustomAppenders();
    Trace::resetPersistantCustomAppenders();
    Trace::disableLogging();

#pragma omp parallel for schedule(dynamic, 1)
    for (unsigned int i = 0; i < nbThreads; i++) {
      launchSimuLocale(jobsFileName);
    }
  } catch (const DYN::Error& e) {
    std::cerr << "DYN Error: " << e.what() << std::endl;
    return e.type();
  } catch (const po::error&) {
    usage(desc);
    return -1;
  } catch (const char* s) {
    std::cerr << "Throws string: '" << s << "'" << std::endl;
    return -1;
  } catch (const string& s) {
    std::cerr << "Throws string: '" << s << "'" << std::endl;
    return -1;
  } catch (const xml::sax::parser::ParserException& exp) {
    std::cerr << DYNLog(XmlParsingError, jobsFileName, exp.what()) << std::endl;
    Trace::error() << DYNLog(XmlParsingError, jobsFileName, exp.what()) << Trace::endline;
    return -1;
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << DYNLog(UnexpectedError) << std::endl;
    Trace::error() << __FILE__ << " " << __LINE__ << " " << DYNLog(UnexpectedError) << Trace::endline;
    return -1;
  }
  return 0;
}

void launchSimuLocale(const std::string& jobsFileName) {
#if defined(_DEBUG_) || defined(PRINT_TIMERS)
  DYN::Timer timer("Main::LaunchSimu");
#endif

  job::XmlImporter importer;
  boost::shared_ptr<job::JobsCollection> jobsCollection = importer.importFromFile(jobsFileName);
  std::string prefixJobFile = absolute(remove_file_name(jobsFileName));
  if (jobsCollection->begin() == jobsCollection->end())
    throw DYNError(DYN::Error::SIMULATION, NoJobDefined);

  for (job::job_iterator itJobEntry = jobsCollection->begin();
       itJobEntry != jobsCollection->end();
       ++itJobEntry) {
    print(DYNLog(LaunchingJob, (*itJobEntry)->getName()));

    boost::shared_ptr<DYN::SimulationContext> context = boost::shared_ptr<DYN::SimulationContext>(new DYN::SimulationContext());
    context->setResourcesDirectory(getMandatoryEnvVar("DYNAWO_RESOURCES_DIR"));
    context->setLocale(getMandatoryEnvVar("DYNAWO_LOCALE"));
    context->setInputDirectory(prefixJobFile);
    context->setWorkingDirectory(prefixJobFile);
    int rank = omp_get_thread_num();
    (*itJobEntry)->getOutputsEntry()->setOutputsDirectory((*itJobEntry)->getOutputsEntry()->getOutputsDirectory() + boost::lexical_cast<std::string>(rank));
    (*itJobEntry)->getModelerEntry()->setCompileDir((*itJobEntry)->getModelerEntry()->getCompileDir() + boost::lexical_cast<std::string>(rank));

    boost::shared_ptr<DYN::Simulation> simulation;
    try {
      simulation = boost::shared_ptr<DYN::Simulation>(new DYN::Simulation((*itJobEntry), context));
      simulation->init();
    } catch (const DYN::Error& err) {
      print(err.what(), DYN::ERROR);
      throw;
    } catch (const DYN::MessageError& e) {
      print(e.what(), DYN::ERROR);
      throw;
    } catch (const char *s) {
      print(s, DYN::ERROR);
      throw;
    } catch (const std::string& Msg) {
      print(Msg, DYN::ERROR);
      throw;
    } catch (const std::exception& exc) {
      print(exc.what(), DYN::ERROR);
      throw;
    }

    try {
      simulation->simulate();
      simulation->terminate();
    } catch (const DYN::Error& err) {
      // Needed as otherwise terminate might crash due to missing staticRef variables
      if (err.key() == DYN::KeyError_t::StateVariableNoReference) {
        simulation->activateExportIIDM(false);
        simulation->setLostEquipmentsExportMode(DYN::Simulation::EXPORT_LOSTEQUIPMENTS_NONE);
      }
      print(err.what(), DYN::ERROR);
      simulation->terminate();
      throw;
    } catch (const DYN::Terminate& e) {
      print(e.what(), DYN::ERROR);
      simulation->terminate();
      throw;
    } catch (const DYN::MessageError& e) {
      print(e.what(), DYN::ERROR);
      simulation->terminate();
      throw;
    } catch (const char *s) {
      print(s, DYN::ERROR);
      simulation->terminate();
      throw;
    } catch (const std::string& Msg) {
      print(Msg, DYN::ERROR);
      simulation->terminate();
      throw;
    } catch (const std::exception& exc) {
      print(exc.what(), DYN::ERROR);
      simulation->terminate();
      throw;
    }
    simulation->clean();
    print(DYNLog(EndOfJob, (*itJobEntry)->getName()));
    // Trace::resetCustomAppenders();
    // Trace::init();
    print(DYNLog(JobSuccess, (*itJobEntry)->getName()));
    if ((*itJobEntry)->getOutputsEntry()) {
      std::string outputsDirectory = createAbsolutePath((*itJobEntry)->getOutputsEntry()->getOutputsDirectory(), context->getWorkingDirectory());
      print(DYNLog(ResultFolder, outputsDirectory));
    }
  }
}